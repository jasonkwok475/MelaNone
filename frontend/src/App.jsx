import { useState, useEffect } from 'react';
import './App.css';
import WelcomeScreen from './pages/WelcomeScreen';
import LoadingScreen from './pages/LoadingScreen';
import ResultsScreen from './pages/ResultsScreen';

function App() {
  const [currentPage, setCurrentPage] = useState('welcome');
  const [analysisData, setAnalysisData] = useState(null);

  const handleStartAnalysis = async () => {
    setCurrentPage('loading');

    // TODO: Replace with actual backend API call
    // const response = await fetch('http://localhost:5000/api/start-scan');
    // const data = await response.json();

    // Simulate backend processing
    setTimeout(() => {
      setAnalysisData({
        totalObjectsAnalyzed: 24,
        concerningSpots: 3
      });
      setCurrentPage('results');
    }, 3000);
  };

  const handleNewAnalysis = () => {
    setCurrentPage('welcome');
    setAnalysisData(null);
  };

  return (
    <div className="app">
      {currentPage === 'welcome' && <WelcomeScreen onStart={handleStartAnalysis} />}
      {currentPage === 'loading' && <LoadingScreen />}
      {currentPage === 'results' && (
        <ResultsScreen data={analysisData} onNewAnalysis={handleNewAnalysis} />
      )}
    </div>
  );
}

export default App;
