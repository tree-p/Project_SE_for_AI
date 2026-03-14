export async function analyzeBeef(imageFile) {

  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await fetch("https://project-se-for-ai.onrender.com/analyze", {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    throw new Error("API error");
  }

  return await response.json();
}