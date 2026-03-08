# ──────────────────────────────────────────────────────────────
# Frontend Dockerfile — React app with multi-stage build
# ──────────────────────────────────────────────────────────────

# ── Stage 1: Build ──
FROM node:20 AS build

WORKDIR /app

# Install dependencies first (layer caching)
COPY package.json package-lock.json ./
RUN npm install

# Build arg for API URL (defaults to /api for nginx proxying)
ARG REACT_APP_API_BASE=http://backend:8000
ENV REACT_APP_API_BASE=$REACT_APP_API_BASE

# Copy source and build
COPY public/ ./public/
COPY src/ ./src/
RUN npm run build

# ── Stage 2: Serve with nginx ──
FROM nginx:1.27-alpine AS production

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets from build stage
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
