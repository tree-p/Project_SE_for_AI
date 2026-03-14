import { useState } from "react";
import Processing from "./Processing";
import Result from "./Result";
import { analyzeBeef } from "../api";

export default function Home() {

  const [image, setImage] = useState(null);
  const [stage, setStage] = useState("upload");
  const [result, setResult] = useState(null);

  const handleSelect = (e) => {
    const file = e.target.files[0];
    if (file) setImage(file);
  };

  const handleAnalyze = async () => {
    setStage("processing");

    try {
      console.log("Sending image to API...");

      const data = await analyzeBeef(image);
      
      console.log("API RESPONSE:", data);

      setResult(data);

      setStage("result");
      
    } catch (err) {
      console.error(err);
      setStage("upload");
    }
  };

  if (stage === "processing") {
    return (
      <div className="phone">
        <Processing />
      </div>
    );
  }

  if (stage === "result") {
    return (
      <div className="phone">
        <Result
          image={image}
          result={result}
          onBack={() => setStage("upload")}
        />
      </div>
    );
  }

  return (
    <div className="phone">
      <h1>Beef Freshness Check</h1>

      <input
        type="file"
        accept="image/*"
        onChange={handleSelect}
      />

      {image && (
        <img
          src={URL.createObjectURL(image)}
          alt="preview"
          width="100%"
        />
      )}

      <button
        disabled={!image}
        onClick={handleAnalyze}
      >
        Analyze Freshness
      </button>
    </div>
  );
}
