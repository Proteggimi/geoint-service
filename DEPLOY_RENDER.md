# Deploy GEOINT Service su Render.com

## Prerequisiti

1. Account GitHub con il repository pushato
2. Account Render.com (gratuito, no carta di credito)

## Step 1: Preparare il Repository

Il backend deve essere in un repository GitHub. Opzioni:

### Opzione A: Repository Separato (Consigliato)
```bash
# Crea nuovo repository solo per il backend
cd geoint-service
git init
git add .
git commit -m "Initial commit - GEOINT Service"
git remote add origin https://github.com/TUO_USERNAME/geoint-service.git
git push -u origin main
```

### Opzione B: Monorepo con path
Se usi lo stesso repo del frontend, Render può buildare da una sottocartella.

## Step 2: Creare Account Render

1. Vai su https://render.com
2. Clicca "Get Started for Free"
3. Registrati con GitHub (consigliato per autorizzazione automatica)

## Step 3: Deploy con render.yaml (Blueprint)

1. In Render Dashboard, clicca "New" → "Blueprint"
2. Connetti il repository GitHub
3. Render rileverà automaticamente `render.yaml`
4. Clicca "Apply" per creare i servizi

Questo creerà:
- **geoint-service**: Web service Python
- **geoint-db**: Database PostgreSQL

## Step 4: Deploy Manuale (Alternativa)

Se preferisci configurare manualmente:

### Creare Database PostgreSQL
1. Dashboard → "New" → "PostgreSQL"
2. Name: `geoint-db`
3. Region: Frankfurt (EU)
4. Plan: Free
5. Clicca "Create Database"
6. Copia la "Internal Database URL"

### Creare Web Service
1. Dashboard → "New" → "Web Service"
2. Connetti repository GitHub
3. Configura:
   - Name: `geoint-service`
   - Region: Frankfurt
   - Branch: main
   - Root Directory: `geoint-service` (se monorepo)
   - Runtime: Python 3
   - Build Command: `pip install -r requirements-render.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. Environment Variables:
   ```
   LITE_MODE=true
   DEBUG=true
   CORS_ORIGINS=https://test.osintnews.it,https://desk.osintnews.it
   DATABASE_URL=[Incolla Internal Database URL]
   ```

5. Clicca "Create Web Service"

## Step 5: Verificare Deploy

1. Attendi che il build completi (2-5 minuti)
2. Clicca sull'URL del servizio (es: `https://geoint-service.onrender.com`)
3. Verifica la risposta:
   ```json
   {
     "service": "GEOINT Service",
     "version": "1.0.0",
     "mode": "lite",
     "docs": "/docs",
     "endpoints": {...}
   }
   ```

4. Testa gli endpoint mock:
   - `GET /health` - Health check
   - `GET /docs` - Swagger UI
   - `GET /mock/scenes?lat=44.6&lon=33.5` - Mock scenes
   - `GET /mock/locations` - Test locations

## Step 6: Configurare Frontend

Aggiorna il frontend per usare il nuovo backend:

```typescript
// In geointService.ts o .env
const GEOINT_API_URL = 'https://geoint-service.onrender.com';
```

## Note Importanti

### Free Tier Limits
- **Sleep**: Il servizio va in sleep dopo 15 min di inattività
- **Startup**: Prima richiesta dopo sleep richiede ~30 secondi
- **RAM**: 512MB (sufficiente per lite mode)
- **Database**: 256MB, 1 database

### Upgrade (Opzionale)
Per funzionalità complete (ML, PostGIS, etc):
- Upgrade a piano Starter ($7/mese)
- Aggiungi Redis service
- Abilita PostGIS manualmente nel database

### PostGIS (se necessario)
Dopo la creazione del database, esegui via psql:
```sql
CREATE EXTENSION IF NOT EXISTS postgis;
```

## Troubleshooting

### Build Fallisce
- Verifica che `requirements-render.txt` esista
- Controlla i log nel dashboard Render

### Database Connection Error
- Verifica che DATABASE_URL sia impostato
- Il formato deve essere: `postgresql://user:pass@host:port/db`

### CORS Errors
- Verifica che CORS_ORIGINS includa il tuo dominio frontend
- Formato: `https://domain1.com,https://domain2.com`

### Service Always Sleeping
- Normale per free tier
- Considera un cron job esterno per keep-alive
- Oppure upgrade a piano paid

## URLs Finali

Dopo il deploy avrai:
- **API**: `https://geoint-service.onrender.com`
- **Docs**: `https://geoint-service.onrender.com/docs`
- **Health**: `https://geoint-service.onrender.com/health`
