import '../styles/WelcomeScreen.css';

export default function WelcomeScreen({ onStart, apiUrl }) {
  const handleStartClick = async () => {
    try {
      console.log('Starting analysis with API URL:', apiUrl);
      
      // Call backend to start analysis
      const response = await fetch(`${apiUrl}/api/start-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Analysis started:', data);
        onStart();
      } else {
        const error = await response.json();
        console.error('Backend error:', error);
        alert('Failed to start analysis: ' + (error.error || 'Unknown error'));
      }
    } catch (error) {
      console.error('Error starting analysis:', error);
      alert('Error connecting to backend: ' + error.message);
    }
  };

  return (
    <div className="welcome-container">
      <div className="welcome-content">
        <h1 className="welcome-title">MelaNone</h1>
        <p className="welcome-description">
          Melanoma detection and analysis system. Press start once your hand is placed in the analysis chamber.
        </p>
        <button className="welcome-button" onClick={handleStartClick}>
          Start Analysis
        </button>
      </div>
    </div>
  );
}
