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

TRAINED_MODEL_DIR = Path("app/resources/trained_models")
SYNTHETIC_DATA_PATH = Path("app/resources/synthetic_strava_data.csv")

@st.cache_resource
def load_model_artifacts():
    artifacts = {}
    
    with open(TRAINED_MODEL_DIR / "modelcard.json", "r") as f:
        artifacts["modelcard"] = json.load(f)
    
    with open(TRAINED_MODEL_DIR / "inference_config.json", "r") as f:
        artifacts["inference_config"] = json.load(f)
    
    artifacts["embeddings"] = np.load(TRAINED_MODEL_DIR / "retrieval/route_embeddings.npy")
    
    with open(TRAINED_MODEL_DIR / "retrieval/route_id_to_idx.json", "r") as f:
        artifacts["route_id_to_idx"] = json.load(f)
    
    with open(TRAINED_MODEL_DIR / "retrieval/feature_columns.json", "r") as f:
        artifacts["feature_columns"] = json.load(f)
    
    artifacts["popularity"] = pd.read_csv(TRAINED_MODEL_DIR / "heuristics/popularity.csv")
    artifacts["route_meta"] = pd.read_csv(TRAINED_MODEL_DIR / "meta/route_meta.csv")
    artifacts["user_seen"] = pd.read_csv(TRAINED_MODEL_DIR / "meta/user_seen.csv")
    
    with open(TRAINED_MODEL_DIR / "heuristics/mmr_config.json", "r") as f:
        artifacts["mmr_config"] = json.load(f)
    
    idx_to_route_id = {idx: route_id for route_id, idx in artifacts["route_id_to_idx"].items()}
    artifacts["idx_to_route_id"] = idx_to_route_id
    
    return artifacts

@st.cache_data
def load_synthetic_data():
    return pd.read_csv(SYNTHETIC_DATA_PATH)

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

def get_recommendations(route_id, artifacts, strategy="content", lambda_param=0.3, k=10):
    route_id_to_idx = artifacts["route_id_to_idx"]
    idx_to_route_id = artifacts["idx_to_route_id"]
    embeddings = artifacts["embeddings"]
    
    if route_id not in route_id_to_idx:
        return None, "Route not found in index"
    
    query_idx = route_id_to_idx[route_id]
    
    if strategy == "content":
        similar_indices, similar_scores = compute_similarity(query_idx, embeddings, k=k)
        results = [(idx_to_route_id[idx], score) for idx, score in zip(similar_indices, similar_scores)]
        
    elif strategy == "content_mmr":
        candidate_k = k * 10
        similar_indices, similar_scores = compute_similarity(query_idx, embeddings, k=candidate_k)
        reranked = mmr_rerank(query_idx, embeddings, similar_indices, similar_scores, lambda_param, k)
        results = [(idx_to_route_id[idx], score) for idx, score in reranked]
        
    elif strategy == "popularity":
        popularity_df = artifacts["popularity"].sort_values("popularity_score", ascending=False)
        top_routes = popularity_df.head(k)
        results = list(zip(top_routes["route_id"], top_routes["popularity_score"]))
    
    else:
        return None, f"Unknown strategy: {strategy}"
    
    return results, None

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
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Recommendations",
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
                ["content", "content_mmr", "popularity"],
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
    
    with tab3:
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
    
    with tab4:
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

