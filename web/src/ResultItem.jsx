import { useEffect, useMemo, useRef, useState } from "react";
import './Result.css'

const fallbackImages = [
    "https://images.unsplash.com/photo-1556911220-e15b29be8c8f",
    "https://images.unsplash.com/photo-1466637574441-749b8f19452f",
    "https://images.unsplash.com/photo-1528712306091-ed0763094c98",
    "https://images.unsplash.com/photo-1495521821757-a1efb6729352",
    "https://images.unsplash.com/photo-1627907228175-2bf846a303b4",
    "https://plus.unsplash.com/premium_photo-1663090715010-4f635ac3324c",
    "https://plus.unsplash.com/premium_photo-1682088910175-109857f95093",
    "https://images.unsplash.com/photo-1585735119407-b037131a352b",
    "https://images.unsplash.com/photo-1720694025145-6f05b62fa334",
    "https://images.unsplash.com/photo-1737625854730-56e11fcaff17"
];

const imageAssignment = new Map();
let imageCycle = [];

function shuffleArray(values) {
    const shuffled = [...values];
    for (let i = shuffled.length - 1; i > 0; i -= 1) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
}

function getRandomImageIndex(itemKey) {
    if (imageAssignment.has(itemKey)) {
        return imageAssignment.get(itemKey);
    }

    if (imageCycle.length === 0) {
        imageCycle = shuffleArray([...Array(fallbackImages.length).keys()]);
    }

    const pickedIndex = imageCycle.pop();
    imageAssignment.set(itemKey, pickedIndex);
    return pickedIndex;
}

export default function ResultItem({ item, index, locateRight }) {
    const ref = useRef();
    const [visible, setVisible] = useState(false);

    const parseInstructions = (raw) => {
        if (!raw) return [];
        if (Array.isArray(raw)) return raw;

        try {
            return JSON.parse(raw);
        } catch (_) {
            try {
                const normalized = raw
                    .replace(/([{,]\s*)'([^']+?)'\s*:/g, '$1"$2":')
                    .replace(/:\s*'([^']*)'/g, ': "$1"');
                return JSON.parse(normalized);
            } catch {
                return [];
            }
        }
    };

    const formatViews = (views) => Number(views || 0).toLocaleString("vi-VN");
    const difficultyLabel = item?.difficulty === 1 ? "Dễ" : item?.difficulty === 2 ? "Trung bình" : item?.difficulty === 3 ? "Khó" : "N/A";
    const difficultyTone = item?.difficulty === 1 ? "easy" : item?.difficulty === 2 ? "medium" : item?.difficulty === 3 ? "hard" : "unknown";
    const popularityLabel = item?.popularity || "N/A";
    const popularityNormalized = String(popularityLabel)
        .toLowerCase()
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "");
    const popularityTone = popularityNormalized.includes("cao") || popularityNormalized.includes("high")
        ? "high"
        : popularityNormalized.includes("trung") || popularityNormalized.includes("medium")
            ? "medium"
            : popularityNormalized.includes("thap") || popularityNormalized.includes("low")
                ? "low"
                : "unknown";
    const ingredients = Array.isArray(item?.ingredients) ? item.ingredients : [];
    const instructions = parseInstructions(item?.instructions);
    const previewIngredients = ingredients.slice(0, 8);
    const previewInstructions = instructions;
    const itemKey = item?.recipe_id || item?._id || `${item?.dish_name || "dish"}-${index}`;
    const fallbackImageIndex = useMemo(() => getRandomImageIndex(itemKey), [itemKey]);
    const fallbackImage = fallbackImages[fallbackImageIndex] || fallbackImages[0];


    useEffect(() => {
        const observer = new IntersectionObserver(([entry]) => {
            if (entry.isIntersecting) {
                setVisible(true);
            } else {
                setVisible(false)
            }
        }, { threshold: 0 });


        if (ref.current) {
            observer.observe(ref.current);
        }

        return () => observer.disconnect();
    }, []);

    return (
        <article
            ref={ref}
            className={`result-card fade-item ${visible ? "show" : ""}`}
            style={{ transitionDelay: `${index * 80}ms` }}
            data-align-right={locateRight ? "true" : "false"}
        >
            <div className="result-card__top">
                <div className="result-card__image-wrap">
                    <img
                        className="result-card__image"
                        src={item?.image_url || fallbackImage}
                        alt={item?.dish_name || "Món ăn"}
                        loading="lazy"
                    />
                    <div className="result-card__badges">
                        <span className="result-card__badge">{item?.category || "Món ăn"}</span>
                        <span className={`result-card__badge result-card__badge--difficulty result-card__badge--${difficultyTone}`}>
                            {difficultyLabel}
                        </span>
                    </div>
                </div>

                <div className="result-card__info">
                    <div className="result-card__title-row">
                        <div className="result-card__title-main">
                            <h2>{item?.dish_name || "Công thức món ăn"}</h2>
                            <span className={`result-card__inline-tag result-card__badge--popularity result-card__badge--popularity-${popularityTone}`}>
                                Phổ biến: {popularityLabel}
                            </span>
                        </div>
                        <div className="result-card__views">
                            <span className="material-symbols-outlined">visibility</span>
                            {formatViews(item?.views)}
                        </div>
                    </div>

                    <div className="result-card__meta-grid">
                        <div><span className="material-symbols-outlined">schedule</span>Tổng: {item?.total_time_min ?? "N/A"} phút</div>
                        <div><span className="material-symbols-outlined">timer</span>Chuẩn bị: {item?.prep_time_min ?? "N/A"} phút</div>
                        <div><span className="material-symbols-outlined">cooking</span>Nấu: {item?.cook_time_min ?? "N/A"} phút</div>
                        <div><span className="material-symbols-outlined">restaurant</span>Khẩu phần: {item?.servings_bin || "N/A"}</div>
                    </div>

                    {item?.url ? (
                        <a className="result-card__link" href={item.url} target="_blank" rel="noreferrer">
                            Xem công thức gốc
                        </a>
                    ) : null}
                </div>
            </div>

            <div className="result-card__bottom">
                <div>
                    <h4>Nguyên liệu</h4>
                    <ul>
                        {previewIngredients.length === 0 ? (
                            <li>Chưa có dữ liệu nguyên liệu.</li>
                        ) : (
                            previewIngredients.map((ingredient, ingredientIndex) => (
                                <li key={`${ingredient?.name || "ingredient"}-${ingredientIndex}`}>
                                    {ingredient?.name || "Nguyên liệu"}
                                    {ingredient?.quantity !== undefined && ingredient?.quantity !== null ? `: ${ingredient.quantity}` : ""}
                                    {ingredient?.unit ? ` ${ingredient.unit}` : ""}
                                </li>
                            ))
                        )}
                    </ul>
                </div>

                <div>
                    <h4>Cách làm</h4>
                    <div className="result-card__steps">
                        {previewInstructions.length === 0 ? (
                            <p>Chưa có hướng dẫn chi tiết.</p>
                        ) : (
                            previewInstructions.map((step, stepIndex) => (
                                <div key={`${step?.step_title || "step"}-${stepIndex}`}>
                                    <h5>{stepIndex + 1}. {step?.step_title || "Bước thực hiện"}</h5>
                                    <p>{step?.content || "Đang cập nhật."}</p>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </article>
    );
}