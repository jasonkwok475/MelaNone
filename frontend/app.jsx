const MelaNone = () => {
  const [loading, setLoading] = React.useState(false);
  const [result, setResult] = React.useState(null);
  const fileInputRef = React.useRef(null);

  const handleImageUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('image', file);

      const response = await fetch('/api/classify', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ error: error.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-2xl p-8 max-w-md w-full">
        <h1 className="text-3xl font-bold text-gray-800 mb-2 text-center">MelaNone</h1>
        <p className="text-gray-600 text-center mb-6">AI-powered melanoma detection</p>

        <div className="space-y-4">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="w-full bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 px-4 rounded-lg transition duration-200"
          >
            Upload Image
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="hidden"
          />

          {loading && (
            <div className="text-center py-4">
              <p className="text-gray-600">Analyzing image...</p>
              <div className="mt-2 animate-spin h-5 w-5 border-2 border-blue-500 border-t-transparent rounded-full mx-auto"></div>
            </div>
          )}

          {result && !result.error && (
            <div className="bg-gray-50 p-4 rounded-lg space-y-3">
              <div className={`p-3 rounded-lg ${result.melanoma ? 'bg-red-100' : 'bg-green-100'}`}>
                <p className={`font-semibold ${result.melanoma ? 'text-red-800' : 'text-green-800'}`}>
                  {result.melanoma ? '⚠️ Melanoma Detected' : '✅ Benign'}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="bg-green-50 p-2 rounded">
                  <p className="text-gray-600">Benign</p>
                  <p className="font-bold text-green-600">{result.benign_confidence * 100}%</p>
                </div>
                <div className="bg-red-50 p-2 rounded">
                  <p className="text-gray-600">Melanoma</p>
                  <p className="font-bold text-red-600">{result.melanoma_confidence * 100}%</p>
                </div>
              </div>

              <p className="text-xs text-gray-500 text-center mt-2">
                ⚠️ Not for medical diagnosis. Consult a dermatologist.
              </p>
            </div>
          )}

          {result?.error && (
            <div className="bg-red-50 p-4 rounded-lg">
              <p className="text-red-800 text-sm">Error: {result.error}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};