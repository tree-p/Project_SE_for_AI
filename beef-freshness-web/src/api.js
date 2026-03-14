export async function analyzeBeef(imageFile) {

  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await fetch("http://127.0.0.1:8000/analyze", {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    throw new Error("API error");
  }

  return await response.json();
}