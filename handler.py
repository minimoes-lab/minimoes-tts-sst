"""
RunPod Serverless Handler
Adapte l'application FastAPI pour RunPod Serverless
"""
import runpod
import json
import asyncio
from api import app
from fastapi.testclient import TestClient

# Client de test pour simuler les requêtes HTTP
client = TestClient(app)

def handler(event):
    """
    Handler principal pour RunPod Serverless
    
    Args:
        event: Événement RunPod contenant la requête
        
    Returns:
        dict: Réponse formatée pour RunPod
    """
    try:
        # Extraire les données de la requête
        input_data = event.get("input", {})
        
        # Déterminer l'endpoint à appeler
        endpoint = input_data.get("endpoint", "/")
        method = input_data.get("method", "GET").upper()
        data = input_data.get("data", {})
        
        # Faire la requête vers l'application FastAPI
        if method == "POST":
            if endpoint == "/upload":
                # Gérer l'upload de fichiers
                files = input_data.get("files", {})
                response = client.post(endpoint, files=files, data=data)
            else:
                response = client.post(endpoint, json=data)
        elif method == "GET":
            response = client.get(endpoint, params=data)
        else:
            return {
                "error": f"Méthode HTTP {method} non supportée",
                "status_code": 405
            }
        
        # Retourner la réponse
        return {
            "status_code": response.status_code,
            "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
            "headers": dict(response.headers)
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status_code": 500
        }

# Démarrer le handler RunPod
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
