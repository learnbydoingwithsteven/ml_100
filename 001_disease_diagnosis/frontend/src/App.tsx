import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:9001';

interface PatientData {
  body_temperature: number;
  blood_pressure: number;
  heart_rate: number;
  glucose_level: number;
  cholesterol: number;
  bmi: number;
  age: number;
  symptom_severity: number;
}

interface DiagnosisResult {
  diagnosis: string;
  diagnosis_code: number;
  confidence: number;
  probabilities: Record<string, number>;
  risk_factors: string[];
  recommendations: string[];
  timestamp: string;
}

interface ModelInfo {
  model_type: string;
  features: string[];
  disease_labels: Record<number, string>;
}

function App() {
  const [patientData, setPatientData] = useState<PatientData>({
    body_temperature: 37.0,
    blood_pressure: 120.0,
    heart_rate: 70.0,
    glucose_level: 90.0,
    cholesterol: 180.0,
    bmi: 22.0,
    age: 30.0,
    symptom_severity: 10.0
  });
  
  const [diagnosisResult, setDiagnosisResult] = useState<DiagnosisResult | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'diagnose' | 'info'>('diagnose');

  useEffect(() => {
    loadModelInfo();
  }, []);

  const loadModelInfo = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/v1/model-info`);
      setModelInfo(response.data);
    } catch (error) {
      console.error('Failed to load model info:', error);
    }
  };

  const handleInputChange = (field: keyof PatientData, value: string) => {
    setPatientData(prev => ({
      ...prev,
      [field]: parseFloat(value) || 0
    }));
  };

  const handleDiagnose = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post<DiagnosisResult>(
        `${API_URL}/api/v1/diagnose`,
        patientData
      );
      setDiagnosisResult(response.data);
    } catch (error: any) {
      setError(error.response?.data?.detail || 'Diagnosis failed');
      console.error('Diagnosis failed:', error);
    }
    setLoading(false);
  };

  const getSeverityColor = (code: number) => {
    const colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336'];
    return colors[code] || '#9E9E9E';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return '#4CAF50';
    if (confidence >= 0.6) return '#FFC107';
    return '#FF9800';
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '40px 20px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        background: 'white',
        borderRadius: '20px',
        boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
        overflow: 'hidden'
      }}>
        {/* Header */}
        <div style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          padding: '30px',
          textAlign: 'center'
        }}>
          <h1 style={{ margin: '0 0 10px 0', fontSize: '2.5em' }}>
            🏥 Disease Diagnosis System
          </h1>
          <p style={{ margin: 0, opacity: 0.9 }}>
            AI-Powered Medical Diagnosis Assistant
          </p>
        </div>

        {/* Tabs */}
        <div style={{
          display: 'flex',
          borderBottom: '2px solid #e0e0e0'
        }}>
          <button
            onClick={() => setActiveTab('diagnose')}
            style={{
              flex: 1,
              padding: '15px',
              border: 'none',
              background: activeTab === 'diagnose' ? 'white' : '#f5f5f5',
              borderBottom: activeTab === 'diagnose' ? '3px solid #667eea' : 'none',
              cursor: 'pointer',
              fontSize: '16px',
              fontWeight: 'bold',
              color: activeTab === 'diagnose' ? '#667eea' : '#666'
            }}
          >
            Diagnosis
          </button>
          <button
            onClick={() => setActiveTab('info')}
            style={{
              flex: 1,
              padding: '15px',
              border: 'none',
              background: activeTab === 'info' ? 'white' : '#f5f5f5',
              borderBottom: activeTab === 'info' ? '3px solid #667eea' : 'none',
              cursor: 'pointer',
              fontSize: '16px',
              fontWeight: 'bold',
              color: activeTab === 'info' ? '#667eea' : '#666'
            }}
          >
            Model Info
          </button>
        </div>

        <div style={{ padding: '30px' }}>
          {activeTab === 'diagnose' ? (
            <div>
              {/* Patient Input Form */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                gap: '20px',
                marginBottom: '30px'
              }}>
                {[
                  { key: 'body_temperature', label: '🌡️ Body Temperature (°C)', min: 35, max: 42, step: 0.1 },
                  { key: 'blood_pressure', label: '💉 Blood Pressure (mmHg)', min: 60, max: 200, step: 1 },
                  { key: 'heart_rate', label: '❤️ Heart Rate (BPM)', min: 40, max: 200, step: 1 },
                  { key: 'glucose_level', label: '🩸 Blood Glucose (mg/dL)', min: 50, max: 400, step: 1 },
                  { key: 'cholesterol', label: '💊 Cholesterol (mg/dL)', min: 100, max: 400, step: 1 },
                  { key: 'bmi', label: '⚖️ BMI', min: 10, max: 60, step: 0.1 },
                  { key: 'age', label: '👤 Age', min: 0, max: 120, step: 1 },
                  { key: 'symptom_severity', label: '🔥 Symptom Severity (0-100)', min: 0, max: 100, step: 1 }
                ].map(field => (
                  <div key={field.key} style={{ marginBottom: '10px' }}>
                    <label style={{
                      display: 'block',
                      marginBottom: '8px',
                      fontWeight: '600',
                      color: '#333'
                    }}>
                      {field.label}
                    </label>
                    <input
                      type="number"
                      value={patientData[field.key as keyof PatientData]}
                      onChange={(e) => handleInputChange(field.key as keyof PatientData, e.target.value)}
                      min={field.min}
                      max={field.max}
                      step={field.step}
                      style={{
                        width: '100%',
                        padding: '12px',
                        border: '2px solid #e0e0e0',
                        borderRadius: '8px',
                        fontSize: '16px',
                        boxSizing: 'border-box'
                      }}
                    />
                  </div>
                ))}
              </div>

              <button
                onClick={handleDiagnose}
                disabled={loading}
                style={{
                  width: '100%',
                  padding: '18px',
                  background: loading ? '#ccc' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '12px',
                  fontSize: '18px',
                  fontWeight: 'bold',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                  marginBottom: '20px'
                }}
              >
                {loading ? 'Analyzing...' : '🔬 Run Diagnosis'}
              </button>

              {error && (
                <div style={{
                  padding: '15px',
                  background: '#ffebee',
                  color: '#c62828',
                  borderRadius: '8px',
                  marginBottom: '20px'
                }}>
                  ⚠️ {error}
                </div>
              )}

              {/* Diagnosis Results */}
              {diagnosisResult && (
                <div style={{
                  marginTop: '30px',
                  padding: '30px',
                  background: '#f8f9fa',
                  borderRadius: '12px',
                  border: '2px solid #e0e0e0'
                }}>
                  <h2 style={{
                    margin: '0 0 20px 0',
                    color: '#333',
                    textAlign: 'center'
                  }}>
                    📋 Diagnosis Results
                  </h2>

                  {/* Main Diagnosis */}
                  <div style={{
                    padding: '25px',
                    background: getSeverityColor(diagnosisResult.diagnosis_code),
                    color: 'white',
                    borderRadius: '12px',
                    marginBottom: '20px',
                    textAlign: 'center',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.2)'
                  }}>
                    <h3 style={{ margin: '0 0 10px 0', fontSize: '1.8em' }}>
                      {diagnosisResult.diagnosis}
                    </h3>
                    <p style={{
                      margin: 0,
                      fontSize: '1.2em',
                      opacity: 0.95
                    }}>
                      Confidence: {(diagnosisResult.confidence * 100).toFixed(1)}%
                    </p>
                  </div>

                  {/* Probabilities */}
                  <div style={{ marginBottom: '20px' }}>
                    <h3 style={{ color: '#333', marginBottom: '15px' }}>
                      📊 Probability Distribution
                    </h3>
                    {Object.entries(diagnosisResult.probabilities).map(([label, prob]) => (
                      <div key={label} style={{ marginBottom: '10px' }}>
                        <div style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          marginBottom: '5px'
                        }}>
                          <span style={{ fontWeight: '600' }}>{label}</span>
                          <span>{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div style={{
                          height: '8px',
                          background: '#e0e0e0',
                          borderRadius: '4px',
                          overflow: 'hidden'
                        }}>
                          <div style={{
                            width: `${prob * 100}%`,
                            height: '100%',
                            background: getConfidenceColor(prob),
                            transition: 'width 0.5s'
                          }} />
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Risk Factors */}
                  <div style={{ marginBottom: '20px' }}>
                    <h3 style={{ color: '#333', marginBottom: '15px' }}>
                      ⚠️ Risk Factors
                    </h3>
                    <ul style={{
                      listStyle: 'none',
                      padding: 0,
                      margin: 0
                    }}>
                      {diagnosisResult.risk_factors.map((factor, idx) => (
                        <li key={idx} style={{
                          padding: '10px 15px',
                          background: 'white',
                          marginBottom: '8px',
                          borderRadius: '8px',
                          borderLeft: '4px solid #FF9800'
                        }}>
                          {factor}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Recommendations */}
                  <div>
                    <h3 style={{ color: '#333', marginBottom: '15px' }}>
                      💡 Recommendations
                    </h3>
                    <ul style={{
                      listStyle: 'none',
                      padding: 0,
                      margin: 0
                    }}>
                      {diagnosisResult.recommendations.map((rec, idx) => (
                        <li key={idx} style={{
                          padding: '10px 15px',
                          background: 'white',
                          marginBottom: '8px',
                          borderRadius: '8px',
                          borderLeft: '4px solid #4CAF50'
                        }}>
                          {rec}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div>
              {/* Model Information */}
              {modelInfo && (
                <div>
                  <h2 style={{ color: '#333', marginBottom: '20px' }}>
                    🤖 Model Information
                  </h2>
                  
                  <div style={{
                    padding: '20px',
                    background: '#f8f9fa',
                    borderRadius: '12px',
                    marginBottom: '20px'
                  }}>
                    <p style={{ margin: '10px 0', fontSize: '16px' }}>
                      <strong>Model Type:</strong> {modelInfo.model_type}
                    </p>
                  </div>

                  <div style={{
                    padding: '20px',
                    background: '#f8f9fa',
                    borderRadius: '12px',
                    marginBottom: '20px'
                  }}>
                    <h3 style={{ color: '#333', marginBottom: '15px' }}>
                      Features Used
                    </h3>
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                      gap: '10px'
                    }}>
                      {modelInfo.features.map((feature, idx) => (
                        <div key={idx} style={{
                          padding: '10px',
                          background: 'white',
                          borderRadius: '8px',
                          border: '2px solid #e0e0e0'
                        }}>
                          {feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div style={{
                    padding: '20px',
                    background: '#f8f9fa',
                    borderRadius: '12px'
                  }}>
                    <h3 style={{ color: '#333', marginBottom: '15px' }}>
                      Disease Categories
                    </h3>
                    {Object.entries(modelInfo.disease_labels).map(([code, label]) => (
                      <div key={code} style={{
                        padding: '12px',
                        background: getSeverityColor(parseInt(code)),
                        color: 'white',
                        borderRadius: '8px',
                        marginBottom: '10px',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center'
                      }}>
                        <span style={{ fontWeight: 'bold' }}>{label}</span>
                        <span style={{
                          padding: '4px 12px',
                          background: 'rgba(255,255,255,0.3)',
                          borderRadius: '12px'
                        }}>
                          Level {code}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div style={{
        textAlign: 'center',
        color: 'white',
        marginTop: '30px',
        opacity: 0.8
      }}>
        <p>⚕️ Disease Diagnosis System v2.0 | Powered by Machine Learning</p>
        <p style={{ fontSize: '0.9em' }}>
          ⚠️ This is a demonstration tool. Always consult healthcare professionals for medical advice.
        </p>
      </div>
    </div>
  );
}

export default App;
