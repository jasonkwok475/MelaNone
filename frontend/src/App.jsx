import { useState, useEffect } from 'react';
import './App.css';
import WelcomeScreen from './pages/WelcomeScreen';
import LoadingScreen from './pages/LoadingScreen';
import ResultsScreen from './pages/ResultsScreen';

// Use environment variable or fallback to localhost
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

function App() {
  const [currentPage, setCurrentPage] = useState('welcome');
  const [analysisData, setAnalysisData] = useState(null);

  useEffect(() => {
    if (currentPage === 'loading') {
      // Listen to backend progress via Server-Sent Events
      const eventSource = new EventSource(`${API_URL}/api/progress`);
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('Progress update:', data);
          
          if (data.status === 'complete' && data.data) {
            console.log('Analysis complete, received data:', data.data);
            setAnalysisData(data.data);
            setCurrentPage('results');
            eventSource.close();
          }
        } catch (error) {
          console.error('Error parsing progress:', error);
        }
      };
      
      eventSource.onerror = () => {
        console.error('Server-Sent Events connection error');
        eventSource.close();
        // Fallback: try to fetch results after delay
        setTimeout(() => {
          fetch(`${API_URL}/api/results`)
            .then(res => res.json())
            .then(data => {
              if (data && data.totalObjectsAnalyzed) {
                setAnalysisData(data);
                setCurrentPage('results');
              }
            })
            .catch(err => console.error('Error fetching results:', err));
        }, 1000);
      };
      
      return () => {
        eventSource.close();
      };
    }
  }, [currentPage]);

  const handleStartAnalysis = () => {
    setCurrentPage('loading');
  };

  const handleNewAnalysis = () => {
    setCurrentPage('welcome');
    setAnalysisData(null);
  };

  return (
    <div className="app">
      {currentPage === 'welcome' && <WelcomeScreen onStart={handleStartAnalysis} apiUrl={API_URL} />}
      {currentPage === 'loading' && <LoadingScreen apiUrl={API_URL} />}
      {currentPage === 'results' && (
        <ResultsScreen data={analysisData} onNewAnalysis={handleNewAnalysis} />
      )}
    </div>
  );
}

export default App;
