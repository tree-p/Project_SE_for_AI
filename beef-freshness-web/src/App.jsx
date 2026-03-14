import meat1 from "./assets/beef.jpg";
import meat2 from "./assets/beef2.jpg";
import meat3 from "./assets/beef3.jpg";
import Home from "./pages/Home";

export default function App() {
  return (
    <>
      {/* Floating background */}
      <div className="floating-bg">
        <img src={meat1} className="float-meat" style={{ top: "10%", left: "-40px" }} />
        <img src={meat2} className="float-meat small" style={{ top: "60%", right: "-30px" }} />
        <img src={meat3} className="float-meat" style={{ top: "35%", right: "15%" }} />
      </div>

      <Home />
    </>
  );
}
