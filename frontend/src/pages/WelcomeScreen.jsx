import '../styles/WelcomeScreen.css';

export default function WelcomeScreen({ onStart }) {
  const handleStartClick = () => {
    // TODO: Call backend functions here to start scan
   
    onStart();
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
