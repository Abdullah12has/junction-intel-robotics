# Junction robot dashboard

Customized Physical AI Studio UI, copied from upstream `junction2026` commit `17714488fa9c7d098a056da83f7c220cf983957d`, with our station dashboard and integration changes. Apache-2.0; see [LICENSE](LICENSE).

The original working copy remains outside this repo; use this folder for future development. No backend or robot drivers are duplicated here. See [station instructions](../docs/ROBOT-DASHBOARD.md).

For the existing Mac tunnel on port 3000:

```sh
cd dashboard
npm ci
PUBLIC_API_PROXY_TARGET=http://127.0.0.1:3000 PUBLIC_API_BASE_URL=http://localhost:3001 npm run start -- --host 127.0.0.1 --port 3001
```

If port 3001 is occupied by the original dashboard, stop that frontend deliberately before starting this copy. Do not run competing robot sessions. With the backend directly available, set the proxy target to its loopback address instead.

Open `/projects/<project_id>/dashboard` and select the configured environment. Connecting holds motor torque; follow the station handoff before operating arms. Generated API schema/types are included for reproducible builds; regenerate them together when the backend contract changes.

## Overview

Web interface for:

- **Camera Management** - Configure and preview camera sources
- **Data Collection** - Record and manage demonstration datasets
- **Training** - Launch and monitor policy training jobs
- **Model Management** - Track trained models and deployments
- **Inference** - Perform inference on robots using the trained models.

## Setup

### Prerequisites

- Node.js >=24.2.0 and npm >=11.14.0
- npm or pnpm
- Backend server running (see [Backend README](../vendor/physical-ai-studio/application/backend/README.md))

### Install Dependencies

```bash
npm install
```

## Development

### Start Dev Server

```bash
npm run start
```

UI runs at http://localhost:3000

### Build for Production

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Code Quality

```bash
# Format code
npm run format

# Lint code
npm run lint

# Type check
npm run type-check
```

## Testing

```bash
# Unit tests
npm run test:unit

# Component tests
npm run test:component
```

## Project Structure

```
ui/src/
├── api/          # API client and hooks
├── components/   # Reusable UI components
├── features/     # Feature-specific modules
├── routes/       # Page components
└── assets/       # Static assets
```

## Configuration

### Environment Variables

Create `.env.local` for custom configuration:

| Variable                  | Description                                     | Default                 |
| ------------------------- | ----------------------------------------------- | ----------------------- |
| `PUBLIC_API_BASE_URL`     | Backend API base URL                            | `http://localhost:3000` |
| `PUBLIC_API_PROXY_TARGET` | Backend target for the development `/api` proxy | `http://localhost:7860` |
| `PUBLIC_ENABLE_PLUGINS`   | Enable the plugin management page and nav item  | enabled                 |

### Development Proxy

The dev server proxies `/api` requests to `PUBLIC_API_PROXY_TARGET` (or
`http://localhost:7860` by default):

```typescript
// rsbuild.config.ts
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:7860',  // Default backend server
      changeOrigin: true,
      ws: true,  // WebSocket support
    },
  },
}
```

This allows the UI to make API calls to `/api/*` which are automatically forwarded to the configured backend.

## See Also

- **[Application Overview](../README.md)** - Application components
- **[Backend](../vendor/physical-ai-studio/application/backend/README.md)** - FastAPI backend service
- **[Library](../vendor/physical-ai-studio/library/README.md)** - Python SDK
