import { useState, useEffect } from 'react';
import '../styles/LoadingScreen.css';

export default function LoadingScreen({ apiUrl }) {
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('Initializing...');

  useEffect(() => {
    // Listen to progress updates from backend
    const eventSource = new EventSource(`${apiUrl}/api/progress`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Loading progress:', data);
        
        if (data.status === 'in_progress') {
          setProgress(data.progress);
          setCurrentStep(data.message || data.step);
        }
      } catch (error) {
        console.error('Error parsing progress:', error);
      }
    };

    eventSource.onerror = () => {
      console.error('Progress stream error');
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [apiUrl]);

  return (
    <div className="loading-container">
      <div className="loading-content">
        <div className="spinner"></div>
        <h2>Analyzing your scan...</h2>
        <p className="loading-step">{currentStep}</p>
        
        <div className="progress-bar-container">
          <div className="progress-bar-fill" style={{ width: `${progress}%` }}></div>
        </div>
        <p className="progress-text">{progress}%</p>
      </div>
    </div>
  );
}
