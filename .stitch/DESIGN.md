# Design System: SentinelFlow
**Project ID:** 5459250966208180226

## 1. Visual Theme & Atmosphere
Futuristic, data-driven command center aesthetic. Dark mode with neon accents. High-tech, cyber vibe with glassmorphism effects. Dense information display with generous breathing room between sections.

## 2. Color Palette & Roles
- **Deep Space** (#0a0e1a) – Page background, canvas
- **Dark Surface** (#111827) – Card backgrounds, elevated surfaces
- **Electric Cyan** (#00d4ff) – Primary accent, links, key data highlights
- **Amber Alert** (#f59e0b) – Warning states, medium-risk indicators
- **Emerald Signal** (#10b981) – Success states, healthy routes
- **Red Alert** (#ef4444) – Critical alerts, high-risk indicators
- **Pure White** (#ffffff) – Primary text, headings
- **Muted Light** (#94a3b8) – Secondary text, labels
- **Glass Border** (rgba(255,255,255,0.1)) – Card borders, dividers

## 3. Typography Rules
- **Font Family:** Inter (Google Fonts)
- **Headings:** Bold/Semibold, tracking tight
- **Body:** Regular 400, 16px base
- **Data Labels:** Medium 500, 14px
- **Monospace Data:** JetBrains Mono for numerical data

## 4. Component Stylings
* **Buttons:** Pill-shaped primary (Electric Cyan bg), ghost secondary (transparent + cyan border)
* **Cards:** Glassmorphism — semi-transparent (#111827/80%), backdrop-blur, thin white/10% border, 8px rounded corners, whisper-soft shadow
* **Inputs:** Dark surface bg, muted border, cyan focus ring
* **Badges/Tags:** Small pill-shaped indicators with status colors
* **Charts:** Cyan/Emerald/Amber color coding, dark grid lines

## 5. Layout Principles
- Max-width container (1400px) with sidebar navigation
- 24px grid gap between cards
- Generous whitespace (32px section padding)
- Sidebar: 260px fixed, collapsible

## 6. Design System Notes for Stitch Generation
**Copy this block into every baton prompt:**

**DESIGN SYSTEM (REQUIRED):**
- Platform: Web, Desktop-first
- Theme: Dark mode, futuristic tech/cyber aesthetic, data-driven command center
- Background: Deep space navy (#0a0e1a), card surfaces (#111827)
- Primary Accent: Electric Cyan (#00d4ff) for links, highlights, active states
- Secondary Accents: Amber (#f59e0b) warnings, Emerald (#10b981) success, Red (#ef4444) alerts
- Text Primary: White (#ffffff), Secondary: Muted (#94a3b8)
- Font: Inter (clean sans-serif), JetBrains Mono for data
- Shape: Softly rounded (8px corners), pill-shaped buttons/badges
- Effects: Glassmorphism cards (semi-transparent + backdrop-blur), subtle glow on accents
- Layout: Sidebar + main content, max-width 1400px, generous whitespace
