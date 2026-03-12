// server.js
// A simple Express server to act as a secure proxy for API requests.

const express = require('express');
const axios = require('axios');
const cors = require('cors');
require('dotenv').config(); // This loads the environment variables from a .env file

const app = express();
const PORT = process.env.PORT || 3001;

// --- Middleware ---
// Enable CORS (Cross-Origin Resource Sharing) so our React app can talk to this server
app.use(cors());
// Parse JSON bodies for incoming requests
app.use(express.json());


// --- API Credentials ---
// Get your BigCommerce API credentials from the .env file.
const BC_STORE_HASH = process.env.BC_STORE_HASH;
const BC_ACCESS_TOKEN = process.env.BC_ACCESS_TOKEN;

if (!BC_STORE_HASH || !BC_ACCESS_TOKEN) {
    console.error("BigCommerce API credentials not found in .env file.");
    console.error("Please create a .env file with BC_STORE_HASH and BC_ACCESS_TOKEN.");
}

const bigCommerceApi = axios.create({
    baseURL: `https://api.bigcommerce.com/stores/${BC_STORE_HASH}/v3`,
    headers: {
        'X-Auth-Token': BC_ACCESS_TOKEN,
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
});


// --- API Routes ---

app.get('/', (req, res) => {
    res.send('Marketing Dashboard Backend is running!');
});

/**
 * @route   GET /api/inventory
 * @desc    Get all product and inventory data from BigCommerce, handling pagination.
 * @access  Public (proxied)
 */
app.get('/api/inventory', async (req, res) => {
    if (!BC_STORE_HASH || !BC_ACCESS_TOKEN) {
        return res.status(500).json({ error: 'Server is missing BigCommerce API credentials.' });
    }

    let allProducts = [];
    let page = 1;
    const limit = 250; // Fetch the maximum number of products per page
    let hasMorePages = true;

    try {
        // Loop through all pages of products from the BigCommerce API
        while (hasMorePages) {
            console.log(`Fetching page ${page} of products...`);
            const response = await bigCommerceApi.get(`/catalog/products?include=variants&limit=${limit}&page=${page}`);
            
            if (response.data.data && response.data.data.length > 0) {
                allProducts = allProducts.concat(response.data.data);
            }

            // Check if there are more pages to fetch
            const meta = response.data.meta;
            if (meta.pagination.current_page >= meta.pagination.total_pages) {
                hasMorePages = false;
            } else {
                page++;
            }
        }
        console.log(`Total products fetched: ${allProducts.length}`);

        // --- Data Transformation ---
        const formattedData = allProducts.map(product => {
            const totalStock = product.variants.reduce((sum, variant) => sum + variant.inventory_level, 0);
            
            let status = 'Out of Stock';
            if (totalStock > 10) {
                status = 'In Stock';
            } else if (totalStock > 0) {
                status = 'Low Stock';
            }

            return {
                id: product.id,
                name: product.name,
                sku: product.sku,
                price: product.price,
                stock: totalStock,
                status: status,
            };
        });

        res.json(formattedData);

    } catch (error) {
        console.error('Error fetching from BigCommerce API:', error.response ? error.response.data : error.message);
        res.status(500).json({ error: 'Failed to fetch data from BigCommerce' });
    }
});


// --- Start the Server ---
app.listen(PORT, () => {
    console.log(`✅ Server is running on http://localhost:${PORT}`);
});
