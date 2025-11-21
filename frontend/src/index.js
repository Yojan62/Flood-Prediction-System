import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import './index.css';
import './App.css'
import App from './App';
import "./styles/globals.css";
import reportWebVitals from './reportWebVitals';

const root = ReactDOM.createRoot(document.getElementById('root'));

// OPTIONAL: Add flags to silence future warnings
const routerFutureFlags = {
  v7_startTransition: true,
  v7_relativeSplatPath: true
};

root.render(
  <React.StrictMode>
    <BrowserRouter future={routerFutureFlags}>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);

reportWebVitals();