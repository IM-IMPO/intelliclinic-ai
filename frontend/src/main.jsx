import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router-dom'
import './styles.css'
import App from './App.jsx'
import { AuthProvider } from './auth.jsx'

// HashRouter, not BrowserRouter: the app is served from a static PWA bundle and
// from a WebView inside the APK, where server-side route rewrites are not
// available, so every link works from a cold load.
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <HashRouter>
      <AuthProvider>
        <App />
      </AuthProvider>
    </HashRouter>
  </StrictMode>,
)
