# OCR Endpoint – Paralelizado por Página

API HTTP para procesar páginas específicas de PDFs almacenados en Google Cloud Storage (GCS) usando OCR con Gemini API. El endpoint procesa múltiples páginas en paralelo y optimiza costos mediante deduplicación.

## Características

- ✅ Procesamiento paralelo de múltiples páginas
- ✅ Deduplicación automática (misma página procesada solo una vez)
- ✅ Rate limiting para respetar cuotas de Gemini API
- ✅ Retry con exponential backoff para errores
- ✅ Arquitectura escalable y modular
- ✅ Health checks integrados

## Requisitos

- Python 3.10+
- Google Cloud SDK (para autenticación con GCS)
- Gemini API Key

## Instalación

1. **Clonar o navegar al directorio del proyecto:**
   ```bash
   cd /Users/sergiocanales/workspace/scratch/smr/ocr
   ```

2. **Crear entorno virtual (recomendado):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno:**
   
   Crear archivo `.env` en la raíz del proyecto:
   ```bash
   # Google Cloud
   # Opción 1: Usar archivo de credenciales (recomendado para desarrollo local)
   GOOGLE_APPLICATION_CREDENTIALS=/ruta/absoluta/a/bucket_credentials.json
   # Ejemplo: GOOGLE_APPLICATION_CREDENTIALS=/Users/sergiocanales/workspace/scratch/smr/ocr/bucket_credentials.json
   
   # Opción 2: Usar credenciales por defecto de gcloud
   # gcloud auth application-default login

   # Gemini API
   GEMINI_API_KEY=your_gemini_api_key_here

   # Rate Limiting
   MAX_CONCURRENT_REQUESTS=10
   GEMINI_RATE_LIMIT_PER_MINUTE=60
   GEMINI_MAX_RETRIES=3
   GEMINI_RETRY_BACKOFF_BASE=2

   # App Config
   ENVIRONMENT=local
   LOG_LEVEL=INFO
   API_V1_PREFIX=/api
   ```

## Ejecución Local

### Opción 1: Script helper (recomendado)
```bash
python scripts/run_local.py
```

### Opción 2: Uvicorn directo
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Opción 3: Python module
```bash
python -m uvicorn app.main:app --reload
```

El servidor estará disponible en: `http://localhost:8000`

## Verificación

- **Health check:** `curl http://localhost:8000/health`
- **Documentación API:** `http://localhost:8000/docs`
- **Endpoint OCR:** `POST http://localhost:8000/api/v1/ocr/pages`

## Uso del Endpoint

### Request

```bash
curl -X POST "http://localhost:8000/api/v1/ocr/pages" \
  -H "Content-Type: application/json" \
  -d '{
    "pages": [
      {
        "bucket": "bucket-escrituras-smr",
        "file": "documento.pdf",
        "page": "1"
      }
    ]
  }'
```

### Response

```json
{
  "results": [
    {
      "page": "1",
      "texto_ocr": "Texto extraído de la página..."
    }
  ]
}
```

## Estructura del Proyecto

```
ocr-endpoint/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app principal
│   ├── config.py                  # Configuración
│   ├── api/                       # Routers y endpoints
│   │   ├── router.py
│   │   └── endpoints/
│   │       └── ocr.py
│   ├── models/                    # Modelos Pydantic
│   │   ├── ocr.py
│   │   └── common.py
│   ├── services/                  # Lógica de negocio
│   │   ├── ocr_service.py
│   │   └── health_service.py
│   ├── clients/                   # Clientes externos
│   │   ├── gcs_client.py
│   │   └── gemini_client.py
│   ├── utils/                     # Utilidades
│   │   ├── pdf_utils.py
│   │   ├── rate_limiter.py
│   │   └── deduplication.py
│   └── core/                      # Core functionality
│       ├── exceptions.py
│       └── logging_config.py
├── scripts/
│   └── run_local.py
├── tests/
├── requirements.txt
├── .env
└── README.md
```

## Características Técnicas

### Deduplicación

El endpoint identifica automáticamente páginas duplicadas (mismo bucket, file, page) y las procesa solo una vez, optimizando costos y tiempo.

**Ejemplo:**
- Input: 100 páginas, 30 duplicadas
- Procesamiento: Solo 70 páginas únicas
- Ahorro: 30% en costos y tiempo

### Paralelización

- Procesa múltiples páginas en paralelo usando `asyncio`
- Control de concurrencia configurable (`MAX_CONCURRENT_REQUESTS`)
- Rate limiting para respetar cuotas de Gemini API

### Rate Limiting

- Token bucket algorithm para limitar requests por minuto
- Retry automático con exponential backoff para errores 429
- Soporte opcional para múltiples proyectos (round-robin)

## Desarrollo

### Linting y Formateo

```bash
# Instalar herramientas de desarrollo (opcional)
pip install black ruff

# Formatear código
black app/

# Linting
ruff check app/
```

### Testing

```bash
# Ejecutar tests (cuando estén implementados)
pytest tests/
```

## Troubleshooting

### Error: "GCS client not initialized"
- Verifica que `GOOGLE_APPLICATION_CREDENTIALS` esté configurado correctamente
- O ejecuta: `gcloud auth application-default login`

### Error: "No Gemini API keys configured"
- Verifica que `GEMINI_API_KEY` esté en el archivo `.env`

### Error: Rate limit exceeded
- Reduce `MAX_CONCURRENT_REQUESTS` o `GEMINI_RATE_LIMIT_PER_MINUTE`
- Considera solicitar aumento de cuota en Google Cloud

## Próximos Pasos (Fase 2)

Una vez que el endpoint funcione correctamente localmente, se procederá con:
- Dockerfile para containerización
- Deploy a Google Cloud Run
- Configuración de Secret Manager
- Scripts de deployment

## Licencia

Este proyecto es privado.

