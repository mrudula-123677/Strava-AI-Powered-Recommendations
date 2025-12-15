import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Strava Route Recommender",
    page_icon="🏃",
    layout="wide"
)

REPO_ROOT = Path(__file__).resolve().parent
TRAINED_MODEL_DIR = REPO_ROOT / "app" / "resources" / "trained_models"
SYNTHETIC_DATA_PATH = REPO_ROOT / "app" / "resources" / "synthetic_strava_data.csv"

@st.cache_resource
def load_model_artifacts():
    artifacts = {}
    
    with open(TRAINED_MODEL_DIR / "modelcard.json", "r", encoding="utf-8") as f:
        artifacts["modelcard"] = json.load(f)
    
    with open(TRAINED_MODEL_DIR / "inference_config.json", "r", encoding="utf-8") as f:
        artifacts["inference_config"] = json.load(f)
    
    artifacts["embeddings"] = np.load(str(TRAINED_MODEL_DIR / "retrieval" / "route_embeddings.npy"))
    
    with open(TRAINED_MODEL_DIR / "retrieval" / "route_id_to_idx.json", "r", encoding="utf-8") as f:
        artifacts["route_id_to_idx"] = json.load(f)
    
    with open(TRAINED_MODEL_DIR / "retrieval" / "feature_columns.json", "r", encoding="utf-8") as f:
        artifacts["feature_columns"] = json.load(f)
    
    artifacts["popularity"] = pd.read_csv(str(TRAINED_MODEL_DIR / "heuristics" / "popularity.csv"))
    artifacts["route_meta"] = pd.read_csv(str(TRAINED_MODEL_DIR / "meta" / "route_meta.csv"))
    artifacts["user_seen"] = pd.read_csv(str(TRAINED_MODEL_DIR / "meta" / "user_seen.csv"))
    
    with open(TRAINED_MODEL_DIR / "heuristics" / "mmr_config.json", "r", encoding="utf-8") as f:
        artifacts["mmr_config"] = json.load(f)
    
    idx_to_route_id = {idx: route_id for route_id, idx in artifacts["route_id_to_idx"].items()}
    artifacts["idx_to_route_id"] = idx_to_route_id
    
    return artifacts

@st.cache_data
def load_synthetic_data():
    return pd.read_csv(str(SYNTHETIC_DATA_PATH))

@st.cache_data
def get_demo_users():
    df = load_synthetic_data()
    if "distance_km_user" in df.columns:
        df["distance_m"] = df["distance_km_user"] * 1000
    elif "distance_m" not in df.columns:
        df["distance_m"] = 5000
    
    if "average_pace_min_per_km" in df.columns and "distance_km_user" in df.columns:
        df["duration_s"] = df["average_pace_min_per_km"] * df["distance_km_user"] * 60
    elif "duration_s" not in df.columns:
        df["duration_s"] = 1800
    
    user_stats = df.groupby('user_id', as_index=False).agg({
        'route_id': 'count',
        'distance_m': 'sum',
        'duration_s': 'sum'
    })
    user_stats.columns = ['user_id', 'activity_count', 'total_distance', 'total_duration']
    return user_stats.sort_values('activity_count', ascending=False)

def get_user_activities(user_id):
    df = load_synthetic_data()
    user_df = df[df['user_id'] == user_id].copy()
    
    if "distance_km_user" in user_df.columns:
        user_df["distance_m"] = user_df["distance_km_user"] * 1000
    if "elevation_meters_user" in user_df.columns:
        user_df["elevation_gain_m"] = user_df["elevation_meters_user"]
    if "average_pace_min_per_km" in user_df.columns and "distance_km_user" in user_df.columns:
        user_df["duration_s"] = user_df["average_pace_min_per_km"] * user_df["distance_km_user"] * 60
    
    if "route_id" in user_df.columns:
        user_df["id"] = user_df["user_id"].astype(str) + "_" + user_df["route_id"].astype(str)
    
    for col in ['distance_m', 'duration_s', 'elevation_gain_m']:
        if col not in user_df.columns:
            user_df[col] = 0.0
    
    user_df = user_df.fillna(0.0)
    
    sport_mapping = {
        'Road': 'running',
        'Trail': 'hiking',
        'Track': 'running',
        'Mixed': 'cycling'
    }
    
    if 'surface_type_route' in user_df.columns:
        user_df['sport'] = user_df['surface_type_route'].map(sport_mapping).fillna('running')
    else:
        user_df['sport'] = 'running'
    
    return user_df

def get_user_seen_routes(user_id, artifacts):
    user_seen_df = artifacts["user_seen"]
    user_routes = user_seen_df[user_seen_df['user_id'] == user_id]['route_id'].tolist()
    return set(str(r) for r in user_routes)

def compute_similarity(query_idx, embeddings, k=10):
    query_vector = embeddings[query_idx:query_idx+1]
    similarities = cosine_similarity(query_vector, embeddings)[0]
    
    similar_indices = np.argsort(-similarities)[1:k+1]
    similar_scores = similarities[similar_indices]
    
    return similar_indices, similar_scores

def mmr_rerank(query_idx, embeddings, candidate_indices, candidate_scores, lambda_param=0.3, k=10):
    selected = []
    selected_indices = []
    remaining = list(zip(candidate_indices, candidate_scores))
    
    query_vector = embeddings[query_idx]
    
    while len(selected) < k and remaining:
        mmr_scores = []
        for idx, base_score in remaining:
            relevance = base_score
            
            if not selected_indices:
                diversity = 0
            else:
                selected_vectors = embeddings[selected_indices]
                candidate_vector = embeddings[idx]
                similarities = cosine_similarity(
                    candidate_vector.reshape(1, -1),
                    selected_vectors
                )[0]
                diversity = np.max(similarities)
            
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append(mmr_score)
        
        best_idx = np.argmax(mmr_scores)
        selected_route_idx, selected_score = remaining.pop(best_idx)
        selected.append((selected_route_idx, selected_score))
        selected_indices.append(selected_route_idx)
    
    return selected

def compute_collaborative_scores(route_id, artifacts):
    user_seen_df = artifacts["user_seen"]
    
    users_with_route = user_seen_df[user_seen_df['route_id'] == route_id]['user_id'].unique()
    
    if len(users_with_route) == 0:
        return {}
    
    other_routes = user_seen_df[user_seen_df['user_id'].isin(users_with_route)]
    route_counts = other_routes[other_routes['route_id'] != route_id]['route_id'].value_counts()
    
    total = len(users_with_route)
    collab_scores = {str(rid): count / total for rid, count in route_counts.items()}
    
    return collab_scores

def get_recommendations(route_id, artifacts, strategy="content", lambda_param=0.3, k=10, user_id=None, exclude_seen=False):
    route_id_to_idx = artifacts["route_id_to_idx"]
    idx_to_route_id = artifacts["idx_to_route_id"]
    embeddings = artifacts["embeddings"]
    
    if route_id not in route_id_to_idx:
        return None, "Route not found in index"
    
    query_idx = route_id_to_idx[route_id]
    
    if strategy == "content":
        similar_indices, similar_scores = compute_similarity(query_idx, embeddings, k=k*2)
        results = [(idx_to_route_id[idx], score) for idx, score in zip(similar_indices, similar_scores)]
        
    elif strategy == "content_mmr":
        candidate_k = k * 10
        similar_indices, similar_scores = compute_similarity(query_idx, embeddings, k=candidate_k)
        reranked = mmr_rerank(query_idx, embeddings, similar_indices, similar_scores, lambda_param, k*2)
        results = [(idx_to_route_id[idx], score) for idx, score in reranked]
        
    elif strategy == "ensemble":
        candidate_k = k * 10
        similar_indices, similar_scores = compute_similarity(query_idx, embeddings, k=candidate_k)
        
        collab_scores = compute_collaborative_scores(route_id, artifacts)
        
        if not collab_scores:
            results = [(idx_to_route_id[idx], score) for idx, score in zip(similar_indices[:k*2], similar_scores[:k*2])]
        else:
            content_normalized = (similar_scores - similar_scores.min()) / (similar_scores.max() - similar_scores.min() + 1e-10)
            
            ensemble_results = []
            for idx, content_score, norm_score in zip(similar_indices, similar_scores, content_normalized):
                rid = idx_to_route_id[idx]
                collab_score = collab_scores.get(rid, 0)
                
                collab_max = max(collab_scores.values()) if collab_scores else 1.0
                collab_normalized = collab_score / (collab_max + 1e-10)
                
                ensemble_score = 0.6 * norm_score + 0.4 * collab_normalized
                ensemble_results.append((rid, ensemble_score))
            
            ensemble_results.sort(key=lambda x: x[1], reverse=True)
            results = ensemble_results[:k*2]
    
    elif strategy == "ensemble_mmr":
        candidate_k = k * 10
        similar_indices, similar_scores = compute_similarity(query_idx, embeddings, k=candidate_k)
        
        collab_scores = compute_collaborative_scores(route_id, artifacts)
        
        if not collab_scores:
            reranked = mmr_rerank(query_idx, embeddings, similar_indices, similar_scores, lambda_param, k*2)
            results = [(idx_to_route_id[idx], score) for idx, score in reranked]
        else:
            content_normalized = (similar_scores - similar_scores.min()) / (similar_scores.max() - similar_scores.min() + 1e-10)
            
            ensemble_scores = []
            for idx, norm_score in zip(similar_indices, content_normalized):
                rid = idx_to_route_id[idx]
                collab_score = collab_scores.get(rid, 0)
                
                collab_max = max(collab_scores.values())
                collab_normalized = collab_score / (collab_max + 1e-10)
                
                ensemble_score = 0.6 * norm_score + 0.4 * collab_normalized
                ensemble_scores.append(ensemble_score)
            
            ensemble_scores = np.array(ensemble_scores)
            reranked = mmr_rerank(query_idx, embeddings, similar_indices, ensemble_scores, lambda_param, k*2)
            results = [(idx_to_route_id[idx], score) for idx, score in reranked]
        
    elif strategy == "popularity":
        popularity_df = artifacts["popularity"].sort_values("popularity_score", ascending=False)
        top_routes = popularity_df.head(k*2)
        results = list(zip(top_routes["route_id"], top_routes["popularity_score"]))
    
    else:
        return None, f"Unknown strategy: {strategy}"
    
    if exclude_seen and user_id:
        seen_routes = get_user_seen_routes(user_id, artifacts)
        results = [(rid, score) for rid, score in results if rid not in seen_routes]
    
    return results[:k], None

def get_route_details(route_id, artifacts):
    route_meta = artifacts["route_meta"]
    route_info = route_meta[route_meta["route_id"] == route_id]
    
    if len(route_info) == 0:
        return None
    
    return route_info.iloc[0].to_dict()

def main():
    st.title("Strava Route Recommender System")
    st.markdown("Production-ready recommendation engine with multiple strategies")
    
    artifacts = load_model_artifacts()
    df = load_synthetic_data()
    
    st.sidebar.header("Model Information")
    modelcard = artifacts["modelcard"]
    
    st.sidebar.metric("Model Version", modelcard["version"])
    st.sidebar.metric("Total Routes", modelcard["training_data"]["num_routes"])
    st.sidebar.metric("Total Users", modelcard["training_data"]["num_users"])
    st.sidebar.metric("Total Activities", modelcard["training_data"]["num_activities"])
    
    st.sidebar.markdown("---")
    st.sidebar.header("Evaluation Metrics")
    metrics = modelcard["evaluation_metrics"]
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Recall@10", f"{metrics['recall_at_10']:.3f}")
        st.metric("MAP@10", f"{metrics['map_at_10']:.4f}")
    with col2:
        st.metric("NDCG@10", f"{metrics['ndcg_at_10']:.4f}")
        st.metric("Recall@5", f"{metrics['recall_at_5']:.3f}")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Recommendations",
        "Live Demo",
        "Strategy Comparison",
        "Data Exploration",
        "Model Details"
    ])
    
    with tab1:
        st.header("Generate Recommendations")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            available_routes = sorted(artifacts["route_id_to_idx"].keys())
            selected_route = st.selectbox(
                "Select a route to get recommendations for:",
                available_routes,
                index=0
            )
        
        with col2:
            k_value = st.slider("Number of recommendations", 5, 20, 10)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strategy = st.selectbox(
                "Recommendation Strategy",
                ["content", "content_mmr", "ensemble", "ensemble_mmr", "popularity"],
                index=1
            )
        
        with col2:
            lambda_param = st.slider(
                "Diversity Parameter (λ)",
                0.0, 1.0, 0.3, 0.05,
                disabled=(strategy != "content_mmr"),
                help="Higher values = more diversity"
            )
        
        with col3:
            st.write("")
            generate_btn = st.button("Generate Recommendations", type="primary")
        
        if generate_btn or "recommendations" not in st.session_state:
            with st.spinner("Computing recommendations..."):
                results, error = get_recommendations(
                    selected_route, 
                    artifacts, 
                    strategy=strategy,
                    lambda_param=lambda_param,
                    k=k_value
                )
                
                if error:
                    st.error(error)
                else:
                    st.session_state.recommendations = results
                    st.session_state.query_route = selected_route
        
        if "recommendations" in st.session_state:
            st.subheader(f"Recommendations for Route: {st.session_state.query_route}")
            
            query_details = get_route_details(st.session_state.query_route, artifacts)
            if query_details:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Distance", f"{query_details['distance_km_route']:.2f} km")
                with col2:
                    st.metric("Elevation", f"{query_details['elevation_meters_route']:.0f} m")
                with col3:
                    st.metric("Difficulty", f"{query_details['difficulty_score']:.2f}")
                with col4:
                    st.metric("Surface", query_details['surface_type_route'])
            
            st.markdown("---")
            
            recommendations_data = []
            for rank, (route_id, score) in enumerate(st.session_state.recommendations, 1):
                details = get_route_details(route_id, artifacts)
                if details:
                    recommendations_data.append({
                        "Rank": rank,
                        "Route ID": route_id,
                        "Score": f"{score:.4f}",
                        "Distance (km)": f"{details['distance_km_route']:.2f}",
                        "Elevation (m)": f"{details['elevation_meters_route']:.0f}",
                        "Difficulty": f"{details['difficulty_score']:.2f}",
                        "Surface": details['surface_type_route']
                    })
            
            rec_df = pd.DataFrame(recommendations_data)
            st.dataframe(rec_df, use_container_width=True, hide_index=True)
            
            fig = px.bar(
                rec_df,
                x="Route ID",
                y=[float(x) for x in rec_df["Score"]],
                title="Recommendation Scores",
                labels={"y": "Similarity Score", "x": "Route ID"}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Live Demo Mode")
        st.markdown("Load synthetic user data and test recommendations interactively")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("User Selection")
            
            demo_users = get_demo_users()
            
            user_options = [
                f"{row['user_id']} ({row['activity_count']} activities, {row['total_distance']/1000:.1f}km)"
                for _, row in demo_users.head(20).iterrows()
            ]
            
            selected_user_display = st.selectbox(
                "Select Demo User",
                user_options,
                index=0
            )
            
            selected_user_id = selected_user_display.split(" (")[0]
            
            col_a, col_b, col_c = st.columns(3)
            user_data = demo_users[demo_users['user_id'] == selected_user_id].iloc[0]
            with col_a:
                st.metric("Activities", user_data['activity_count'])
            with col_b:
                st.metric("Distance", f"{user_data['total_distance']/1000:.1f}km")
            with col_c:
                st.metric("Time", f"{user_data['total_duration']/3600:.1f}h")
            
            st.markdown("---")
            
            st.subheader("Strategy Settings")
            
            demo_strategy = st.selectbox(
                "Recommendation Strategy",
                ["content", "content_mmr", "ensemble", "ensemble_mmr", "popularity"],
                index=1,
                key="demo_strategy"
            )
            
            strategy_descriptions = {
                "content": "Fast cosine similarity matching",
                "content_mmr": "Best quality: Balances relevance with diversity using MMR",
                "ensemble": "Combines content-based + collaborative filtering",
                "ensemble_mmr": "Best coverage: Ensemble with diversity reranking",
                "popularity": "Shows most popular routes from historical data"
            }
            
            st.info(strategy_descriptions[demo_strategy])
            
            if "mmr" in demo_strategy:
                demo_lambda = st.slider(
                    "Diversity Level",
                    0.0, 1.0, 0.3, 0.1,
                    key="demo_lambda",
                    help="Higher values = more diversity"
                )
                
                if demo_lambda <= 0.3:
                    st.caption("Similar to usual routes")
                elif demo_lambda < 0.7:
                    st.caption("Mix of familiar and new routes")
                else:
                    st.caption("Discover different route types")
            else:
                demo_lambda = 0.3
            
            demo_exclude_seen = st.checkbox(
                "Exclude routes already completed",
                value=False,
                key="demo_exclude_seen"
            )
            
            demo_k = st.slider("Number of recommendations", 5, 20, 10, key="demo_k")
        
        with col2:
            st.subheader("User Activities")
            
            user_activities = get_user_activities(selected_user_id)
            
            if len(user_activities) > 0:
                st.info(f"Showing {len(user_activities)} activities for {selected_user_id}")
                
                activity_display = []
                for _, activity in user_activities.head(50).iterrows():
                    route_id = activity.get('route_id', 'Unknown')
                    activity_display.append({
                        "Route": route_id,
                        "Sport": activity.get('sport', 'running').title(),
                        "Distance": f"{activity['distance_m']/1000:.2f} km",
                        "Duration": f"{activity['duration_s']/60:.0f} min",
                        "Elevation": f"{activity.get('elevation_gain_m', 0):.0f} m"
                    })
                
                activity_df = pd.DataFrame(activity_display)
                
                selected_activity_idx = st.selectbox(
                    "Select an activity to get recommendations",
                    range(len(activity_display)),
                    format_func=lambda i: f"{activity_display[i]['Route']} - {activity_display[i]['Sport']} - {activity_display[i]['Distance']}"
                )
                
                selected_activity_data = user_activities.iloc[selected_activity_idx]
                selected_route_id = str(selected_activity_data.get('route_id', ''))
                
                st.markdown("---")
                
                col_x, col_y, col_z = st.columns(3)
                with col_x:
                    st.metric("Distance", f"{selected_activity_data['distance_m']/1000:.2f} km")
                with col_y:
                    st.metric("Elevation", f"{selected_activity_data.get('elevation_gain_m', 0):.0f} m")
                with col_z:
                    st.metric("Duration", f"{selected_activity_data['duration_s']/60:.0f} min")
                
                if st.button("Generate Recommendations", type="primary", key="demo_generate"):
                    with st.spinner("Computing recommendations..."):
                        demo_results, demo_error = get_recommendations(
                            selected_route_id,
                            artifacts,
                            strategy=demo_strategy,
                            lambda_param=demo_lambda,
                            k=demo_k,
                            user_id=selected_user_id,
                            exclude_seen=demo_exclude_seen
                        )
                        
                        if demo_error:
                            st.error(demo_error)
                        else:
                            st.session_state.demo_recommendations = demo_results
                            st.session_state.demo_query_route = selected_route_id
                            st.session_state.demo_user_id = selected_user_id
                
                if "demo_recommendations" in st.session_state:
                    st.markdown("---")
                    st.subheader(f"Recommendations for Route: {st.session_state.demo_query_route}")
                    
                    if demo_exclude_seen:
                        seen_count = len(get_user_seen_routes(st.session_state.demo_user_id, artifacts))
                        st.caption(f"Filtered out {seen_count} routes already completed by this user")
                    
                    demo_rec_data = []
                    for rank, (route_id, score) in enumerate(st.session_state.demo_recommendations, 1):
                        details = get_route_details(route_id, artifacts)
                        if details:
                            demo_rec_data.append({
                                "Rank": rank,
                                "Route": route_id,
                                "Score": f"{score:.4f}",
                                "Distance": f"{details['distance_km_route']:.2f} km",
                                "Elevation": f"{details['elevation_meters_route']:.0f} m",
                                "Difficulty": f"{details['difficulty_score']:.2f}",
                                "Surface": details['surface_type_route']
                            })
                    
                    demo_rec_df = pd.DataFrame(demo_rec_data)
                    st.dataframe(demo_rec_df, use_container_width=True, hide_index=True)
                    
                    fig = px.bar(
                        demo_rec_df,
                        x="Route",
                        y=[float(x) for x in demo_rec_df["Score"]],
                        title=f"Recommendation Scores ({demo_strategy})",
                        labels={"y": "Similarity Score", "x": "Route ID"},
                        color=[float(x) for x in demo_rec_df["Score"]],
                        color_continuous_scale="viridis"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption(f"Strategy: {demo_strategy} | Lambda: {demo_lambda if 'mmr' in demo_strategy else 'N/A'}")
            else:
                st.warning("No activities found for this user")
    
    with tab3:
        st.header("Strategy Comparison")
        
        strategies_info = {
            "content": {
                "name": "Pure Similarity",
                "description": "Baseline content-based filtering using feature similarity",
                "speed": "Fastest",
                "quality": "Baseline"
            },
            "content_mmr": {
                "name": "Content + Diversity (MMR)",
                "description": "Content-based with MMR reranking for diverse results",
                "speed": "Fast",
                "quality": "Best MAP & NDCG"
            },
            "ensemble": {
                "name": "Ensemble (Content + Collaborative)",
                "description": "Combines content-based and collaborative filtering (60/40 blend)",
                "speed": "Moderate",
                "quality": "Improved coverage"
            },
            "ensemble_mmr": {
                "name": "Ensemble + Diversity",
                "description": "Ensemble approach with MMR reranking for maximum coverage",
                "speed": "Moderate",
                "quality": "Best Recall"
            },
            "popularity": {
                "name": "Popularity-Based",
                "description": "Recommends popular routes based on usage frequency",
                "speed": "Fastest",
                "quality": "Cold-start friendly"
            }
        }
        
        for strategy_key, info in strategies_info.items():
            with st.expander(f"{info['name']} ({strategy_key})"):
                st.markdown(f"**Description:** {info['description']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Speed:** {info['speed']}")
                with col2:
                    st.markdown(f"**Quality:** {info['quality']}")
        
        st.markdown("---")
        st.subheader("Performance Metrics Comparison")
        
        metrics_df = pd.DataFrame({
            "Strategy": ["content_mmr"],
            "Recall@10": [metrics['recall_at_10']],
            "MAP@10": [metrics['map_at_10']],
            "NDCG@10": [metrics['ndcg_at_10']]
        })
        
        fig = go.Figure()
        for metric in ["Recall@10", "MAP@10", "NDCG@10"]:
            fig.add_trace(go.Bar(
                name=metric,
                x=metrics_df["Strategy"],
                y=metrics_df[metric],
                text=metrics_df[metric].round(4),
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Evaluation Metrics by Strategy",
            xaxis_title="Strategy",
            yaxis_title="Score",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Data Exploration")
        
        st.subheader("Route Statistics")
        route_meta = artifacts["route_meta"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                route_meta,
                x="distance_km_route",
                nbins=30,
                title="Distribution of Route Distances",
                labels={"distance_km_route": "Distance (km)"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                route_meta,
                x="elevation_meters_route",
                nbins=30,
                title="Distribution of Route Elevations",
                labels={"elevation_meters_route": "Elevation (m)"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            surface_counts = route_meta["surface_type_route"].value_counts()
            fig = px.pie(
                values=surface_counts.values,
                names=surface_counts.index,
                title="Route Surface Types"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                route_meta,
                x="distance_km_route",
                y="elevation_meters_route",
                color="difficulty_score",
                title="Distance vs Elevation (colored by difficulty)",
                labels={
                    "distance_km_route": "Distance (km)",
                    "elevation_meters_route": "Elevation (m)",
                    "difficulty_score": "Difficulty"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Popularity Distribution")
        popularity_df = artifacts["popularity"].sort_values("popularity_score", ascending=False).head(20)
        fig = px.bar(
            popularity_df,
            x="route_id",
            y="popularity_score",
            title="Top 20 Most Popular Routes",
            labels={"route_id": "Route ID", "popularity_score": "Popularity Score"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("User Activity Patterns")
        user_seen = artifacts["user_seen"]
        user_activity_counts = user_seen.groupby("user_id").size().reset_index(name="activity_count")
        user_activity_counts = user_activity_counts.sort_values("activity_count", ascending=False).head(20)
        
        fig = px.bar(
            user_activity_counts,
            x="user_id",
            y="activity_count",
            title="Top 20 Most Active Users",
            labels={"user_id": "User ID", "activity_count": "Number of Activities"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("Model Details")
        
        st.subheader("Model Card")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Information**")
            st.json({
                "Model Name": modelcard["model_name"],
                "Version": modelcard["version"],
                "Trained At": modelcard["trained_at"],
                "Best Strategy": modelcard["best_strategy"]
            })
        
        with col2:
            st.markdown("**Training Data**")
            st.json(modelcard["training_data"])
        
        st.subheader("Model Components")
        components = modelcard["model_components"]
        
        components_df = pd.DataFrame({
            "Component": list(components.keys()),
            "Enabled": list(components.values())
        })
        st.dataframe(components_df, use_container_width=True, hide_index=True)
        
        st.subheader("Feature Schema")
        st.markdown(f"**Scaler Type:** {modelcard['feature_schema']['scaler_type']}")
        st.markdown(f"**Number of Features:** {len(modelcard['feature_schema']['columns'])}")
        
        with st.expander("View All Features"):
            features_df = pd.DataFrame({
                "Feature Name": modelcard['feature_schema']['columns']
            })
            st.dataframe(features_df, use_container_width=True, hide_index=True)
        
        st.subheader("Inference Configuration")
        inference_config = artifacts["inference_config"]
        st.json(inference_config)
        
        st.subheader("MMR Configuration")
        mmr_config = artifacts["mmr_config"]
        st.json(mmr_config)
        
        st.subheader("Strategy Descriptions")
        strategy_desc = modelcard["strategy_descriptions"]
        for strategy, description in strategy_desc.items():
            st.markdown(f"**{strategy}:** {description}")

if __name__ == "__main__":
    main()

