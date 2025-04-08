 # start of file
#!/bin/bash

# Script to set up the Next.js frontend for the Key Management Module

# Navigate to the frontend directory
cd "$(dirname "$0")/../frontend" || exit

# Install dependencies
echo "Installing frontend dependencies..."
npm install

# Create necessary environment file
echo "Creating environment file..."
cat > .env.local << EOL
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:3000/api
NEXT_PUBLIC_APP_NAME=Key Management Module
EOL

echo "Frontend setup complete!"
echo "To start the development server, run: cd frontend && npm run dev"
