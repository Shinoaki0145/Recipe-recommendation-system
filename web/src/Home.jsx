import NavBar from './NavBar.jsx'
import SearchInput from './SearchInput.jsx'

export default function Home({ onSearch, isLoading }) {
    return (
        <>
            <NavBar></NavBar>
            <SearchInput onSearch={onSearch} isLoading={isLoading}></SearchInput>
        </>
    )
}