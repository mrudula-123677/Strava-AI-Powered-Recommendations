# Tech Stack we have used and why
## Frontend

- To begin with we have used **React with vite** as our frontend as vite is much faster as compared with the traditional create react app and HMR (Hot Module Replacement) provides instant feedback during developement.
- As we are building a GPS tracking application that needs real-time updates, Vite's real time lightning-fast dev server means we can iterate quickly on map features and see changes instantly
- **TailwindCSS** was chosen over css or JS (Javascript) or plain css is that strava-style apps require clean and mobile-responsive UIs. TailwindCSS let's you build complex layouts (activity cards, map overlays, stat panels) without leaving your JSX
- We made use of **Mapbox GL** which is a professional grade maps with GPU accelerated rendering which shows animations required in applications with ease.
- We also added **Leaflet** as a fallback in case of Mapbox GL is down or not available which is a free open source which uses OpenStreetMap and is also lightweight and not so great but it's just a fall back incases when Mapbox GL isn't available so users won't get problems.

## Backend
- We used **FastAPI** as our backend instead of Flask or Django because fastapi's async/await let's you handle GPS updates from many users while computing recommendations for other users without blocking. As our application contains CPU intensive ML tasks (Recommendtaion engine) and Real time location sharing which would require constant updates.
- We chose to use **PostgreSQL with PostGIS** extension as it turns PostgreSQL into a geospatial powerhouse and other databases don't can't do this efficiently as our application contains actions like distance calculations, route proximity which can be done in the PostgreSQL by taking advantage of the extension PostGIS.
- When our recommendtation engine computes similarity scores between routes, Without caching every recommendation request would a lot of time as it requires loading user interaction matrix, Computing the cosine similarity and then Merge with content-based scores, which is a lot of time. But with **redis**, repeated requests server cached results in much lesser time inturn making the application faster and reduces the load on the backend server too.
- When suggesting users our content-based recommender compares the factors distance, elevation, surface type, historical preferences. So, if we use brute force approach there can be many calculations per query which is resource intensive, this can be evaded with **FAISS** which uses pre-built index and gives out results much faster and makes use of ANN (Approximate Nearest Neighbours) which is higher accuracy and gives our results much faster when compared with the raw brute force approach.

## Security & Authentication
- As we are developing a GPS tracking based application we need persistant login the users mustn't loging every time while they start a run. Hence we are using **JWT with python-jose** instead of the usual sessions. Because JWT provides Access token and Refresh token whose durations can be customised as per our usecase. And an additional main reason being it doesn't require database lookups for every request as they are self-contained.
- An additional security measure to be considered is saving raw passwords in a databse whih is not a suggestable approach so we chose **Bcrypt password hashing** which ensures that passwords can't be crackable as this methods of hashing uses **Salting**.

## ML Architecture
- we used a multi strategy Recommender (Content + Collaborative + MMR) because
    - Content-based: Works for new routes (no user data needed)
    - Collaborative filtering: "Users like you also ran..."
    - MMR reranking: Ensures diverse recommendations (not 10 similar routes)
    - Ensemble: Combines both for better coverage
- The reason being: Amazon uses this exact architecture:
    - Content: "You viewed hiking boots" → Show similar boots
    - Collaborative: "Users who bought this also bought..." → Show popular  items
    - Diversity: Mix hiking boots, tents, and backpacks (not 10 similar boots)

## Deployment strategy
- We made use of docker as it's ease to maintain and ease of scalability and the main reason being it isolates everything so all those services need not be installed in the machine and it works on any machine irrespective of the OS being used.
- Instead of typing long commands as **docker compose up** and other complex commands we have made use of Makefile to simplify the commands. So it's easier and much more understandable for beginner developers who taken on the project as a continuation to add more features.

## Alembic for migrations
- You start with users table. Later you add followers, location_sharing, notifications. Without migrations, you'd need to:
    - Manually write SQL
    - Track which changes were applied
    - Hope everyone runs the same commands
- This whole process can be automated using alembic

- This tech stack was chosen for modern web standards, ML performance, and real-world scalability. Each piece solves a specific problem in building a GPS tracking + recommendation platform.