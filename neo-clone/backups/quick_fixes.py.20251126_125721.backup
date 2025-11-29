#!/usr/bin/env python3
"""
Quick Fix: Improved Site Recognition for Neo-Clone
"""

# Enhanced site patterns for better recognition
ENHANCED_SITE_PATTERNS = {
    'search_engine': {
        'primary_indicators': [
            'google.com', 'bing.com', 'duckduckgo.com', 'yahoo.com/search',
            'search', 'input[name="q"]', 'textarea[name="q"]', '.gLFyf',
            'input[title*="search"]', '[aria-label*="search"]'
        ],
        'secondary_indicators': [
            'search button', 'magnifying glass', 'search results',
            'input[type="search"]', '.search', '#search'
        ],
        'confidence_threshold': 0.6
    },
    
    'ecommerce': {
        'primary_indicators': [
            'amazon.com', 'ebay.com', 'shopify.com', 'walmart.com',
            'cart', 'checkout', 'add to cart', 'buy now', 'price',
            'product', 'shipping', 'payment', 'order'
        ],
        'secondary_indicators': [
            'reviews', 'wishlist', 'compare', 'categories',
            'sale', 'discount', 'customer reviews'
        ],
        'confidence_threshold': 0.7
    },
    
    'social_media': {
        'primary_indicators': [
            'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
            'login', 'signup', 'post', 'share', 'like', 'comment',
            'profile', 'timeline', 'news feed'
        ],
        'secondary_indicators': [
            'followers', 'following', 'friends', 'messages',
            'notifications', 'settings', 'logout'
        ],
        'confidence_threshold': 0.7
    },
    
    'corporate': {
        'primary_indicators': [
            'about us', 'contact', 'services', 'products',
            'company', 'team', 'careers', 'investors',
            'support', 'help', 'documentation'
        ],
        'secondary_indicators': [
            'privacy policy', 'terms of service', 'sitemap',
            'newsletter', 'blog', 'press release'
        ],
        'confidence_threshold': 0.6
    },
    
    'news_media': {
        'primary_indicators': [
            'news', 'article', 'breaking', 'headlines',
            'journalist', 'editor', 'publication', 'media'
        ],
        'secondary_indicators': [
            'category', 'trending', 'popular', 'latest',
            'opinion', 'analysis', 'investigation'
        ],
        'confidence_threshold': 0.6
    }
}

# Multi-selector strategy for robustness
ROBUST_SELECTORS = {
    'search_input': [
        'textarea[name="q"]',      # Google modern
        'input[name="q"]',         # Google fallback
        'input[type="search"]',     # HTML5 search
        'input[title*="Search"]',  # Title-based
        '[aria-label*="search"]',   # Accessibility
        '.search-input',            # Common class
        '#search',                  # Common ID
        '.gLFyf',                 # Google specific
        '#twotabsearchtextbox'      # Amazon specific
    ],
    
    'login_email': [
        'input[type="email"]',
        'input[name="email"]',
        'input[name="username"]',
        'input[id*="email"]',
        'input[placeholder*="email"]',
        '.email-input',
        '#email'
    ],
    
    'login_password': [
        'input[type="password"]',
        'input[name="password"]',
        'input[id*="password"]',
        '.password-input',
        '#password'
    ],
    
    'submit_button': [
        'button[type="submit"]',
        'input[type="submit"]',
        'button:contains("Submit")',
        'button:contains("Login")',
        'button:contains("Sign In")',
        '.submit-btn',
        '#submit'
    ]
}

# Smart retry mechanism
async def smart_retry(page, action_func, max_attempts=3, delay=1):
    """Execute action with intelligent retry."""
    for attempt in range(max_attempts):
        try:
            return await action_func()
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"Action failed after {max_attempts} attempts: {e}")
                raise e
            
            # Wait with exponential backoff
            wait_time = delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
            await page.wait_for_timeout(wait_time * 1000)

# Enhanced site analysis
def analyze_site_enhanced(content, url):
    """Improved site analysis with better pattern matching."""
    content_lower = content.lower()
    url_lower = url.lower()
    
    scores = {}
    
    for site_type, patterns in ENHANCED_SITE_PATTERNS.items():
        score = 0
        
        # Check primary indicators (weighted more heavily)
        for indicator in patterns['primary_indicators']:
            if indicator in content_lower or indicator in url_lower:
                score += 2
        
        # Check secondary indicators (weighted less)
        for indicator in patterns['secondary_indicators']:
            if indicator in content_lower:
                score += 1
        
        # Normalize score
        max_possible = len(patterns['primary_indicators']) * 2 + len(patterns['secondary_indicators'])
        normalized_score = score / max_possible if max_possible > 0 else 0
        
        scores[site_type] = {
            'score': normalized_score,
            'confidence': normalized_score,
            'meets_threshold': normalized_score >= patterns['confidence_threshold']
        }
    
    # Find best match
    best_type = max(scores.keys(), key=lambda k: scores[k]['score'])
    best_score = scores[best_type]
    
    return {
        'site_type': best_type if best_score['meets_threshold'] else 'unknown',
        'confidence': best_score['confidence'],
        'all_scores': scores,
        'url': url
    }

print("Enhanced site recognition patterns loaded!")
print("This improves Neo-Clone's site type detection accuracy.")