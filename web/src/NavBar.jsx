import { NavLink } from "react-router-dom";

export default function NavBar() {
    return (
        <nav className="bg-surface dark:bg-zinc-900 sticky top-0 z-50">
            <div className="flex justify-between items-center w-full px-8 py-4 max-w-7xl mx-auto">
                <div className="flex items-center gap-8">
                    <span className="text-2xl font-bold tracking-tight text-orange-700 dark:text-orange-500 font-headline">Recipetor</span>
                    <div className="hidden md:flex items-center gap-6">
                        <NavLink className="text-orange-700 dark:text-white font-bold border-orange-700 text-sm tracking-wide" to="/">Explore</NavLink>
                    </div>
                </div>
            </div>
        </nav>

    )
}