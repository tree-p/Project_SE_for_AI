import { useRef, useState } from "react";

export default function Result({ image, result, onBack }) {

  const imgRef = useRef(null);
  const [scale, setScale] = useState({x:1,y:1});

  const bbox = result?.roi_bbox;

  const handleImageLoad = () => {

    const img = imgRef.current;

    const scaleX = img.clientWidth / img.naturalWidth;
    const scaleY = img.clientHeight / img.naturalHeight;

    setScale({x:scaleX, y:scaleY});
  };

  return (
    <>
      <h2>Inspection Result</h2>

      <div className="card" style={{ position: "relative" }}>

        <img
          ref={imgRef}
          src={URL.createObjectURL(image)}
          alt="result"
          width="100%"
          style={{ borderRadius: "16px" }}
          onLoad={handleImageLoad}
        />

        {bbox && (
          <div
            style={{
              position: "absolute",
              top: bbox.y * scale.y,
              left: bbox.x * scale.x,
              width: bbox.width * scale.x,
              height: bbox.height * scale.y,
              border: "3px solid #b11226",
              borderRadius: "12px",
              pointerEvents: "none"
            }}
          />
        )}

      </div>

      <div className="result-row">
        <strong>Status:</strong>{" "}
        <span className="badge">
          {result.freshness}
        </span>
      </div>

      <div className="result-row">
        <strong>Confidence:</strong>{" "}
        {(result.confidence.confidence_score * 100).toFixed(1)}%
        {" "}
        ({result.confidence.reliability})
      </div>

      <div className="card">
        <strong>Recommendation</strong>
        <p>{result.advice}</p>
      </div>

      <button style={{ marginTop: "18px" }} onClick={onBack}>
        Check Another Sample
      </button>
    </>
  );
}