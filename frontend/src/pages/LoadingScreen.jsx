import '../styles/LoadingScreen.css';

export default function LoadingScreen() {
  return (
    <div className="loading-container">
      <div className="loading-content">
        <div className="spinner"></div>
        <h2>Analyzing your scan...</h2>
        <p>This may take a moment</p>
      </div>
    </div>
  );
}
