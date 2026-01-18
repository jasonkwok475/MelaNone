from flask import Flask, jsonify, Response, send_from_directory
from flask_cors import CORS
import threading
import json
import os
from backend.api.start import AnalysisManager


"""
Main entrypoint for the serverside 
"""
# Set up paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
FRONTEND_DIST = os.path.join(ROOT_DIR, 'frontend', 'dist')

# Check if frontend is built, otherwise use static folder
if os.path.exists(FRONTEND_DIST):
    app = Flask(__name__, static_folder=FRONTEND_DIST, static_url_path='')
else:
    app = Flask(__name__)

CORS(app)

# Global analysis manager
analysis_manager = AnalysisManager()


@app.route('/')
def index():
    """Serve the React app or development index"""
    if os.path.exists(FRONTEND_DIST):
        return send_from_directory(FRONTEND_DIST, 'index.html')
    return jsonify({'message': 'MelaNone Backend API. Frontend not built. Run: npm run build in frontend folder'})


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from dist folder"""
    if os.path.exists(FRONTEND_DIST):
        file_path = os.path.join(FRONTEND_DIST, path)
        if os.path.isfile(file_path):
            return send_from_directory(FRONTEND_DIST, path)
        return send_from_directory(FRONTEND_DIST, 'index.html')
    return jsonify({'error': 'Frontend not built'}), 404


@app.route('/api/start-analysis', methods=['POST'])
def start_analysis():
    """Start the analysis in a background thread"""
    if analysis_manager.is_running:
        return jsonify({'error': 'Analysis already running'}), 400
    
    # Start analysis in background thread
    thread = threading.Thread(target=analysis_manager.run_analysis)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'Analysis started', 'message': 'Processing your scan...'}), 200


@app.route('/api/progress')
def progress():
    """Stream progress updates via Server-Sent Events"""
    def generate():
        while analysis_manager.is_running or not analysis_manager.queue.empty():
            try:
                if not analysis_manager.queue.empty():
                    data = analysis_manager.queue.get(timeout=1)
                    yield f"data: {json.dumps(data)}\n\n"
                else:
                    yield f": heartbeat\n\n"
            except:
                yield f": heartbeat\n\n"
        
        # Send final completion message
        if analysis_manager.results:
            yield f"data: {json.dumps({'status': 'complete', 'data': analysis_manager.results})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
        'Connection': 'keep-alive'
    })


@app.route('/api/results')
def get_results():
    """Get analysis results"""
    if analysis_manager.results:
        return jsonify(analysis_manager.results), 200
    return jsonify({'error': 'No results available'}), 404


@app.route('/api/status')
def status():
    """Get current analysis status"""
    return jsonify({
        'is_running': analysis_manager.is_running,
        'progress': analysis_manager.progress,
        'current_step': analysis_manager.current_step
    }), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
