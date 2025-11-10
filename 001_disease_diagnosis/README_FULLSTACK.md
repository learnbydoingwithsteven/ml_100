# Disease Diagnosis System - Full-Stack Guide

## Production-Ready Medical Diagnosis Application

### Quick Start with Docker
```bash
# Start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:10001
# Backend API: http://localhost:9001
# API Documentation: http://localhost:9001/docs
```

### Manual Development Setup

#### Backend Setup
```bash
cd backend

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 9001
```

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

### Technology Stack

#### Backend
- **Framework**: FastAPI 0.104.1
- **ML Libraries**: scikit-learn 1.3.0, pandas 2.0.3, numpy 1.24.3
- **Server**: uvicorn with async support
- **Testing**: pytest with 9 comprehensive tests
- **API Documentation**: Auto-generated OpenAPI/Swagger docs

#### Frontend
- **Framework**: React 18+ with TypeScript
- **HTTP Client**: Axios
- **Styling**: Inline styles with modern gradient design
- **Features**: Real-time diagnosis, interactive forms, visual results

#### Infrastructure
- **Containerization**: Docker & Docker Compose
- **Database**: PostgreSQL 15 Alpine (configured)
- **Cache**: Redis 7 Alpine (configured)
- **Ports**:
  - Frontend: 10001
  - Backend: 9001
  - PostgreSQL: 5432 (internal)
  - Redis: 6379 (internal)

### API Endpoints

#### Core Endpoints
- `GET /` - Root endpoint with system info
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `GET /redoc` - Alternative API documentation

#### Diagnosis Endpoints
- `POST /api/v1/diagnose` - Submit patient data for diagnosis
- `POST /api/v1/train` - Retrain the ML model
- `GET /api/v1/model-info` - Get model details
- `GET /api/v1/feature-importance` - Get feature importance rankings

### Features

#### Backend Features
✅ **ML Model Integration**
- RandomForest classifier with 78% accuracy
- Automatic training on startup
- Real-time predictions with confidence scores

✅ **Patient Analysis**
- 8 vital sign/symptom features
- 5-level disease severity classification
- Risk factor identification
- Personalized recommendations

✅ **API Features**
- RESTful architecture
- Input validation with Pydantic
- CORS enabled for frontend
- Comprehensive error handling
- Structured logging

✅ **Testing**
- 9 automated tests
- Edge case coverage
- Input validation tests
- Model performance tests

#### Frontend Features
✅ **User Interface**
- Modern gradient design
- Responsive layout
- Tab-based navigation
- Real-time form validation

✅ **Diagnosis Features**
- Interactive patient data entry
- Visual probability distribution
- Color-coded severity levels
- Risk factor display
- Treatment recommendations

✅ **Model Information**
- Feature list display
- Disease category overview
- Model type and parameters

### Testing

#### Backend Tests
```bash
cd backend
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

**Test Coverage**: 9 tests passing
- Root and health endpoints
- Diagnosis with various patient profiles
- Input validation
- Model training and retraining
- Feature importance extraction
- Edge case handling

#### Manual API Testing
```bash
# Test health endpoint
curl http://localhost:9001/health

# Test diagnosis
curl -X POST http://localhost:9001/api/v1/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "body_temperature": 37.0,
    "blood_pressure": 120.0,
    "heart_rate": 70.0,
    "glucose_level": 90.0,
    "cholesterol": 180.0,
    "bmi": 22.0,
    "age": 30.0,
    "symptom_severity": 10.0
  }'
```

### Environment Variables

Create `.env` file based on `.env.example`:
```bash
# Backend
DATABASE_URL=postgresql://admin:password@postgres:5432/app001
REDIS_URL=redis://redis:6379
PYTHON_ENV=development

# Frontend
REACT_APP_API_URL=http://localhost:9001
```

### Deployment

#### Docker Compose (Recommended)
```bash
# Production build
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### Manual Deployment
1. Build backend Docker image
2. Build frontend Docker image
3. Deploy PostgreSQL and Redis
4. Configure environment variables
5. Start services in order: DB → Redis → Backend → Frontend

### Architecture

```
┌─────────────────┐
│   React Frontend│
│  (Port 10001)   │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│ FastAPI Backend │
│  (Port 9001)    │
├─────────────────┤
│  ML Model       │
│  (RandomForest) │
└────┬───────┬────┘
     │       │
     ▼       ▼
┌─────────┐ ┌─────┐
│PostgreSQL│ │Redis│
└─────────┘ └─────┘
```

### Development Workflow

1. **Backend Development**
   - Modify `backend/app/main.py`
   - Add tests in `backend/tests/`
   - Run tests: `pytest tests/ -v`
   - API auto-reloads with uvicorn

2. **Frontend Development**
   - Modify `frontend/src/App.tsx`
   - Hot reload with `npm start`
   - Build production: `npm run build`

3. **Full Stack Testing**
   - Start backend: `uvicorn app.main:app --reload`
   - Start frontend: `npm start`
   - Test integration

### Troubleshooting

**Backend won't start**
- Check Python version (3.8+)
- Install dependencies: `pip install -r requirements.txt`
- Verify port 9001 is available

**Frontend won't start**
- Check Node version (14+)
- Clear cache: `rm -rf node_modules package-lock.json`
- Reinstall: `npm install`
- Verify port 10001 is available

**API calls failing**
- Check CORS configuration
- Verify REACT_APP_API_URL in frontend
- Check backend logs for errors

**Docker issues**
- Rebuild: `docker-compose up --build`
- Check logs: `docker-compose logs backend`
- Verify ports not in use

### Security Notes

⚠️ **Development Configuration**
- Current setup uses default passwords
- CORS allows all origins
- No authentication/authorization

🔒 **Production Recommendations**
- Use strong, unique passwords
- Configure specific CORS origins
- Implement authentication (JWT)
- Enable HTTPS
- Use environment variable management
- Regular security updates

### Performance

- **Backend**: Async FastAPI handles concurrent requests
- **ML Model**: Trained once on startup, cached in memory
- **Database**: PostgreSQL with connection pooling
- **Cache**: Redis for session/temporary data

### Support

For issues or questions:
- Review API docs: http://localhost:9001/docs
- Check backend logs: `docker-compose logs backend`
- Run tests: `pytest tests/ -v`
- See main README.md for detailed ML information

## License
MIT License

## Version
2.0.0 - Full-Stack Production Release
