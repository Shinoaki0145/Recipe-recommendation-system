import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function SearchInput({ onSearch, isLoading }) {
    const [query, setQuery] = useState("");
    const [limit, setLimit] = useState(5);
    const navigate = useNavigate();

    const handleSearch = async () => {
        if (isLoading) return;
        await onSearch(query, limit);
        navigate("/results");
    };

    const handleKeyDown = (event) => {
        if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault();
            handleSearch();
        }
    };

    return (
        <section className="relative min-h-[700px] flex flex-col items-center justify-center px-6 overflow-hidden">
            <div className="absolute inset-0 z-0 opacity-10 pointer-events-none overflow-hidden">
                <div className="absolute -top-24 -right-24 w-96 h-96 rounded-full bg-primary blur-[120px]" />
                <div className="absolute bottom-0 left-1/4 w-[600px] h-[600px] rounded-full bg-secondary-container blur-[150px]" />
            </div>
            <div className="relative z-10 max-w-4xl w-full text-center space-y-8">
                <div className="space-y-4">
                    <h1 className="font-headline text-5xl md:text-7xl font-extrabold editorial-tight text-on-surface">
                        What are you <span className="text-primary">craving</span>? 
                    </h1>
                    <p className="text-on-surface-variant text-lg md:text-xl max-w-2xl mx-auto font-body leading-relaxed">
                        Describe what you want to cook, the ingredients you have, or a feeling you want to evoke. We'll find the perfect recipe.
                    </p>
                </div>
                <div className="relative group max-w-3xl mx-auto mt-12">
                    <div className="absolute inset-0 bg-primary/10 blur-xl rounded-full opacity-0 group-focus-within:opacity-100 transition-opacity" />
                    <div className="relative flex items-center bg-surface-container-lowest rounded-full shadow-sm p-2 transition-all duration-300 focus-within:shadow-xl">
                        <span className="material-symbols-outlined ml-6 text-outline">restaurant</span>
                        <textarea
                            className="w-full max-h-32 px-6 py-4 bg-transparent border-none focus:ring-0 focus:outline-none text-on-surface text-lg placeholder:text-outline/60 resize-none [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]"
                            placeholder="Tell me what you're craving..."
                            rows="1"
                            value={query}
                            onChange={(event) => setQuery(event.target.value)}
                            onKeyDown={handleKeyDown}
                        ></textarea>
                        <button
                            type="button"
                            className="hero-gradient text-on-primary font-bold px-8 py-4 rounded-full flex items-center gap-2 hover:scale-[1.02] active:scale-95 transition-all disabled:opacity-70 disabled:cursor-not-allowed shrink-0"
                            onClick={handleSearch}
                            disabled={isLoading}
                        >
                            <span>{isLoading ? "Loading..." : "Search"}</span>
                            <span className="material-symbols-outlined text-sm">{isLoading ? "hourglass_top" : "auto_awesome"}</span>
                        </button>
                    </div>
                    
                    <div className="mt-6 flex justify-center items-center gap-3 opacity-80 hover:opacity-100 transition-opacity select-none">
                        <span className="text-on-surface-variant text-sm font-medium">Return up to</span>
                        <div className="flex items-center bg-surface-container-lowest rounded-full p-1 border border-outline-variant/30 shadow-sm">
                            <button 
                                type="button"
                                onMouseDown={(e) => e.preventDefault()}
                                onClick={() => setLimit(prev => Math.max(1, Number(prev || 5) - 1))}
                                className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-surface-container hover:text-primary transition-colors text-on-surface-variant active:scale-95"
                            >
                                <span className="material-symbols-outlined text-sm">remove</span>
                            </button>
                            <input 
                                type="number"
                                min="1"
                                max="30"
                                value={limit} 
                                onChange={(e) => {
                                    let val = e.target.value;
                                    if (val !== "") {
                                        val = parseInt(val, 10);
                                        if (val > 30) val = 30;
                                        if (val < 1) val = 1;
                                    }
                                    setLimit(val);
                                }}
                                onBlur={(e) => {
                                    if (limit === "" || limit < 1) setLimit(5);
                                }}
                                className="bg-transparent text-center w-10 text-on-surface font-bold focus:ring-0 focus:outline-none [-moz-appearance:_textfield] [&::-webkit-outer-spin-button]:m-0 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:m-0 [&::-webkit-inner-spin-button]:appearance-none"
                            />
                            <button 
                                type="button"
                                onMouseDown={(e) => e.preventDefault()}
                                onClick={() => setLimit(prev => Math.min(30, Number(prev || 5) + 1))}
                                className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-surface-container hover:text-primary transition-colors text-on-surface-variant active:scale-95"
                            >
                                <span className="material-symbols-outlined text-sm">add</span>
                            </button>
                        </div>
                        <span className="text-on-surface-variant text-sm font-medium">results</span>
                    </div>
                </div>

            </div>
        </section>
    )
}