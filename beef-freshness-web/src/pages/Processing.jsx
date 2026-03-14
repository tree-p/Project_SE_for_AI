import { useEffect } from "react";

export default function Processing({ onDone }) {
  useEffect(() => {
    const timer = setTimeout(onDone, 2500);
    return () => clearTimeout(timer);
  }, [onDone]);

  return (
    <>
      <h2>Analyzing Sample</h2>
      <p>Please wait while the system evaluates the image</p>

      <ul style={{ fontSize: "14px", color: "#444" }}>
        <li>✔ Image quality verification</li>
        <li>✔ Color & lighting normalization</li>
        <li>✔ Region of interest detection</li>
        <li>✔ Freshness classification</li>
      </ul>
    </>
  );
}
