import './App.css'
import { useState } from 'react';
import Home from './Home';
import { Routes, Route } from "react-router-dom";
import ResultPage from './ResultPage';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

function App() {
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchRecommendations = async (query) => {
    const response = await fetch(`${API_BASE_URL}/rank`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        candidate_top_k: 30,
        result_top_k: 5
      }),
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const data = await response.json();

    return Array.isArray(data?.items) ? data.items : [];
  };

  const handleSearch = async (query) => {
    setIsLoading(true);
    try {
      const apiResults = await fetchRecommendations(query);
      setResults(Array.isArray(apiResults) ? apiResults : []);
    } catch (error) {
      console.error("Failed to fetch recommendations:", error);
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Routes>
      <Route path="/" element={<Home onSearch={handleSearch} isLoading={isLoading} />} />
      <Route path="/results" element={<ResultPage results={results} isLoading={isLoading} />} />
    </Routes>
  );
}

export default App
