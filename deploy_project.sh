#!/bin/bash

# Deploy BTC Price Predictor project

echo "Deploying BTC Price Predictor project..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Docker is installed
if ! command_exists docker; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command_exists docker-compose; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Build Docker images
echo "Building Docker images..."
docker-compose build

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "❌ Docker build failed. Please check the error messages above."
    exit 1
fi

# Deploy options
echo "Select deployment option:"
echo "1. Local deployment (docker-compose)"
echo "2. Kubernetes deployment"
echo "3. Cloud deployment (AWS)"
echo "4. Cloud deployment (GCP)"
echo "5. Cloud deployment (Azure)"
read -p "Enter option (1-5): " deploy_option

case $deploy_option in
    1)
        echo "Starting local deployment with Docker Compose..."
        docker-compose up -d
        
        if [ $? -eq 0 ]; then
            echo "✅ Application deployed successfully!"
            echo "Access the application at:"
            echo "- Frontend: http://localhost:3000"
            echo "- API: http://localhost:8000"
            echo "- API Documentation: http://localhost:8000/docs"
        else
            echo "❌ Deployment failed. Please check the error messages above."
            exit 1
        fi
        ;;
    2)
        echo "Preparing Kubernetes deployment..."
        
        # Check if kubectl is installed
        if ! command_exists kubectl; then
            echo "❌ kubectl is not installed. Please install kubectl first."
            exit 1
        fi
        
        # Check if Kubernetes manifests directory exists
        if [ ! -d "kubernetes" ]; then
            echo "Creating Kubernetes manifests directory..."
            mkdir -p kubernetes
            
            # Create Kubernetes deployment manifest
            cat > kubernetes/deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: btc-predictor-backend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: btc-predictor-backend
  template:
    metadata:
      labels:
        app: btc-predictor-backend
    spec:
      containers:
      - name: backend
        image: btc-predictor-backend:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
        - containerPort: 8765
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: btc-predictor-frontend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: btc-predictor-frontend
  template:
    metadata:
      labels:
        app: btc-predictor-frontend
    spec:
      containers:
      - name: frontend
        image: btc-predictor-frontend:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 3000
EOF

            # Create Kubernetes service manifest
            cat > kubernetes/service.yaml << EOF
apiVersion: v1
kind: Service
metadata:
  name: btc-predictor-backend
spec:
  selector:
    app: btc-predictor-backend
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  - name: websocket
    port: 8765
    targetPort: 8765
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: btc-predictor-frontend
spec:
  selector:
    app: btc-predictor-frontend
  ports:
  - port: 3000
    targetPort: 3000
  type: LoadBalancer
EOF

            # Create Kubernetes ingress manifest
            cat > kubernetes/ingress.yaml << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: btc-predictor-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: btc-predictor-backend
            port:
              number: 8000
      - path: /ws
        pathType: Prefix
        backend:
          service:
            name: btc-predictor-backend
            port:
              number: 8765
      - path: /
        pathType: Prefix
        backend:
          service:
            name: btc-predictor-frontend
            port:
              number: 3000
EOF
        fi
        
        echo "Do you want to deploy to Kubernetes now? (y/n)"
        read -r answer
        if [ "$answer" = "y" ]; then
            echo "Deploying to Kubernetes..."
            kubectl apply -f kubernetes/deployment.yaml
            kubectl apply -f kubernetes/service.yaml
            kubectl apply -f kubernetes/ingress.yaml
            
            echo "✅ Application deployed to Kubernetes!"
            echo "Check the status with: kubectl get pods"
        else
            echo "Kubernetes deployment skipped. You can deploy manually with:"
            echo "kubectl apply -f kubernetes/"
        fi
        ;;
    3)
        echo "AWS deployment selected. This is not implemented yet."
        echo "You can deploy to AWS using:"
        echo "- AWS ECS (Elastic Container Service)"
        echo "- AWS EKS (Elastic Kubernetes Service)"
        echo "- AWS Fargate"
        echo "Please refer to the AWS documentation for more information."
        ;;
    4)
        echo "GCP deployment selected. This is not implemented yet."
        echo "You can deploy to GCP using:"
        echo "- Google Kubernetes Engine (GKE)"
        echo "- Google Cloud Run"
        echo "Please refer to the GCP documentation for more information."
        ;;
    5)
        echo "Azure deployment selected. This is not implemented yet."
        echo "You can deploy to Azure using:"
        echo "- Azure Kubernetes Service (AKS)"
        echo "- Azure Container Instances (ACI)"
        echo "Please refer to the Azure documentation for more information."
        ;;
    *)
        echo "Invalid option selected. Exiting."
        exit 1
        ;;
esac

echo "Deployment process completed!"