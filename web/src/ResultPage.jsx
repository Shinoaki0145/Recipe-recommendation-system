import ResultItem from "./ResultItem";
import "./Result.css";
import NavBar from "./NavBar";

export default function ResultPage({ results = [], isLoading = false }) {
  void items;
  const displayItems = Array.isArray(results) ? results : [];

  return (
    <>
      <NavBar></NavBar>
      <main className="result-page">
        <div className="result-page__container">
          <div className="result-page__header">
            <h1>Recipe Recommendations with your preferences</h1>
            <p>
              {isLoading
                ? "Fetching suitable recipe suggestions..."
                : displayItems.length > 0
                  ? `Found ${displayItems.length} recipe recommendations for you!`
                  : "No suitable results found."}
            </p>
          </div>

          <section className="result-page__list" aria-live="polite">
            {!isLoading && displayItems.length === 0 ? (
              <div className="result-page__empty">
                No suitable results found. Please try describing different ingredients or dishes.
              </div>
            ) : (
              displayItems.map((item, index) => (
                <ResultItem
                  key={item?.recipe_id || item?._id || `${item?.dish_name}-${index}`}
                  item={item}
                  index={index}
                />
              ))
            )}
          </section>
        </div>
      </main>
    </>
  );
}