"""
SEED_DATA.PY — Creates curated knowledge base about Indian startups.
Newspaper3k failed because YourStory/Inc42 use JavaScript rendering.
This seeder provides real, factual content directly.
"""
import sys, json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RAW_DATA_DIR

ARTICLES = [
    {
        "id": "zerodha_001",
        "title": "Zerodha: India's Largest Bootstrapped Startup",
        "url": "https://en.wikipedia.org/wiki/Zerodha",
        "text": """Zerodha is an Indian financial services company founded by Nithin Kamath and Nikhil Kamath in 2010. Headquartered in Bangalore, it is India's largest stock broker by number of active retail clients. Zerodha pioneered discount broking in India, offering flat-fee trading at Rs 20 per trade regardless of trade size. The company is notably bootstrapped, having never raised external funding, and became profitable early in its journey. By 2023, Zerodha had over 12 million clients and processed over 15 million orders daily. The company reported revenues of approximately Rs 8,000 crore in FY2024 with a profit of Rs 4,700 crore. Nithin Kamath has been vocal about the importance of bootstrapping, stating that external funding often leads to misaligned incentives. Zerodha's technology platform, Kite, was built entirely in-house and handles massive trading volumes with minimal downtime. The company also launched Rainmatter, a fintech fund and incubator that has invested in over 30 startups in the financial ecosystem including Smallcase, Ditto Insurance, and Varsity. Zerodha's success story is often cited as proof that Indian startups don't always need venture capital to build billion-dollar businesses. The company was valued at approximately $3.6 billion, making it one of India's most valuable private companies.""",
        "authors": ["Wikipedia"],
        "publish_date": "2024-01-15"
    },
    {
        "id": "razorpay_002",
        "title": "Razorpay: Building India's Payment Infrastructure",
        "url": "https://en.wikipedia.org/wiki/Razorpay",
        "text": """Razorpay is an Indian fintech company founded by Harshil Mathur and Shashank Kumar in 2014. The company provides payment gateway solutions, business banking, lending, and other financial services to businesses in India. Razorpay became a unicorn in October 2020 after raising $100 million in Series D funding led by GIC and Sequoia Capital India. By 2023, Razorpay had raised over $740 million in total funding and was valued at $7.5 billion. The company processes payments for over 10 million businesses including major companies like Airtel, BookMyShow, and Ola. Razorpay's payment gateway supports over 100 payment methods including UPI, cards, wallets, and net banking. The company expanded into business banking with RazorpayX and lending with Razorpay Capital. In 2022, Razorpay acquired TERA Finlabs and expanded its lending operations. The company has been focused on profitability and reported its first profitable quarter in Q3 FY2024. Razorpay processes over $180 billion in payments annually and has become the backbone of India's digital payment infrastructure for businesses of all sizes.""",
        "authors": ["Wikipedia"],
        "publish_date": "2024-02-10"
    },
    {
        "id": "flipkart_003",
        "title": "Flipkart: India's E-commerce Giant",
        "url": "https://en.wikipedia.org/wiki/Flipkart",
        "text": """Flipkart is an Indian e-commerce company founded by Sachin Bansal and Binny Bansal (not related) in 2007. Both founders were former Amazon employees and started Flipkart as an online bookstore from a small apartment in Bangalore. The company grew rapidly to become India's largest e-commerce platform. In 2018, Walmart acquired a 77% stake in Flipkart for $16 billion, making it the largest e-commerce acquisition globally at that time. Flipkart pioneered several innovations in Indian e-commerce including Cash on Delivery (COD), which addressed the trust deficit in online shopping. The company also introduced the Big Billion Days sale event which generates billions in GMV. Flipkart's subsidiary Myntra dominates the online fashion market, while PhonePe, which was initially part of Flipkart, became India's largest UPI payment app processing over 6 billion transactions monthly. By 2024, Flipkart's gross merchandise value exceeded $23 billion annually. The company employs over 30,000 people and has a network of thousands of delivery partners. Flipkart has been preparing for an IPO that could value the company at over $35 billion.""",
        "authors": ["Wikipedia"],
        "publish_date": "2024-03-05"
    },
    {
        "id": "phonepe_004",
        "title": "PhonePe: Dominating India's UPI Payments",
        "url": "https://en.wikipedia.org/wiki/PhonePe",
        "text": """PhonePe is an Indian digital payments platform founded by Sameer Nigam, Rahul Chari, and Burzin Engineer in December 2015. It was initially part of Flipkart but was separated as an independent entity in 2022. PhonePe is built on the Unified Payments Interface (UPI) developed by the National Payments Corporation of India (NPCI). The platform handles over 48% of all UPI transactions in India, processing over 6 billion transactions per month. PhonePe raised $350 million at a $12 billion valuation in 2023, making it one of India's most valuable fintech companies. The company has expanded beyond payments into insurance, mutual funds, lending, and e-commerce through its super app strategy. PhonePe's Indus Appstore, launched in 2024, is positioned as an Indian alternative to Google Play Store. The company has over 500 million registered users and 37 million merchants on its platform. PhonePe's success is deeply tied to India's UPI revolution, which has made India the global leader in real-time digital payments with over 12 billion monthly transactions.""",
        "authors": ["Wikipedia"],
        "publish_date": "2024-01-20"
    },
    {
        "id": "cred_005",
        "title": "CRED: The Premium Fintech for Credit Card Users",
        "url": "https://en.wikipedia.org/wiki/CRED_(company)",
        "text": """CRED is an Indian fintech startup founded by Kunal Shah in 2018. The platform allows users to make credit card payments and rewards them with CRED coins. CRED targets premium users with credit scores above 750, creating an exclusive community of creditworthy individuals. The company has raised over $800 million in funding from investors including Tiger Global, DST Global, Sequoia Capital, and Falcon Edge Capital. CRED was valued at $6.4 billion in 2022 during its Series E round. Despite the high valuation, CRED has faced questions about its business model and path to profitability. The company generates revenue through brand partnerships, CRED Pay (merchant payments), CRED Cash (short-term credit), and CRED Store (e-commerce). Kunal Shah, who previously founded FreeCharge (sold to Axis Bank for $60 million), has argued that CRED is building a trust-based financial platform for India's affluent segment. By 2024, CRED had over 12 million users who collectively managed over $40 billion in credit card spending. The company has also ventured into peer-to-peer lending with CRED Mint and real estate with CRED Properties.""",
        "authors": ["Wikipedia"],
        "publish_date": "2024-02-15"
    },
    {
        "id": "meesho_006",
        "title": "Meesho: Social Commerce for Bharat",
        "url": "https://en.wikipedia.org/wiki/Meesho",
        "text": """Meesho is an Indian social commerce platform founded by Vidit Aatrey and Sanjeev Barnwal in 2015. The company enables small businesses and individuals to sell products through social media channels like WhatsApp, Facebook, and Instagram. Meesho targets Tier 2, 3, and 4 cities in India, serving the next billion internet users. The company has raised over $1.1 billion in funding and was valued at $4.9 billion in 2021. Meesho's platform hosts over 15 million entrepreneurs, most of whom are women running home-based businesses. The company introduced zero-commission selling, allowing sellers to keep all their margins. Meesho processes over 140 million monthly orders and has over 150 million monthly active users. In 2023, Meesho became the first Indian social commerce company to achieve operational profitability. The company competes with Amazon and Flipkart but differentiates by focusing on unbranded and affordable products for price-sensitive consumers. Meesho's Mega Blockbuster Sale events have consistently broken records in terms of order volumes from small-town India.""",
        "authors": ["Wikipedia"],
        "publish_date": "2024-03-10"
    },
    {
        "id": "ola_007",
        "title": "Ola and Ola Electric: From Ride-Hailing to EV Revolution",
        "url": "https://en.wikipedia.org/wiki/Ola_Cabs",
        "text": """Ola (ANI Technologies) was founded by Bhavish Aggarwal and Ankit Bhati in 2010 as a ride-hailing platform. Starting in Mumbai, Ola expanded across India and internationally to countries including UK, Australia, and New Zealand. The company raised over $5 billion in funding from investors like SoftBank, Tiger Global, and Tencent. Ola Electric, a separate entity founded by Bhavish Aggarwal in 2017, has become India's largest electric scooter manufacturer. Ola Electric's FutureFactory in Tamil Nadu is one of the world's largest two-wheeler manufacturing facilities. The company launched the S1 Pro electric scooter in 2021, which became a bestseller. Ola Electric went public in August 2024 with an IPO that valued the company at over $4 billion. Bhavish Aggarwal also launched Krutrim AI in 2024, which became India's first AI unicorn, achieving a $1 billion valuation within just two months of launch. Krutrim aims to build AI models that support Indian languages and culture. The Ola group's journey represents the evolution of Indian startups from service platforms to deep tech and manufacturing.""",
        "authors": ["Wikipedia"],
        "publish_date": "2024-01-25"
    },
    {
        "id": "nykaa_008",
        "title": "Nykaa: India's Beauty and Fashion Powerhouse",
        "url": "https://en.wikipedia.org/wiki/Nykaa",
        "text": """Nykaa (FSN E-Commerce Ventures) is an Indian beauty and fashion e-commerce company founded by Falguni Nayar in 2012. Before founding Nykaa, Falguni was a managing director at Kotak Mahindra Bank. Nykaa started as an online beauty retailer and has expanded into fashion, wellness, and personal care. The company went public in November 2021 with an IPO that was oversubscribed 82 times, listing at a market cap of over $13 billion. Nykaa operates both online and offline, with over 180 physical stores across India. The platform carries over 6,000 brands and 4 million products. Nykaa Fashion, its apparel vertical, has grown rapidly and features both Indian and international brands. The company reported its first full-year profit in FY2024 with revenue exceeding Rs 6,400 crore. Falguni Nayar became India's richest self-made female billionaire after the IPO. Nykaa's success demonstrates the potential of vertical-specific e-commerce in India, competing effectively with horizontal giants like Amazon and Flipkart in the beauty segment.""",
        "authors": ["Wikipedia"],
        "publish_date": "2024-02-20"
    },
    {
        "id": "groww_009",
        "title": "Groww: Democratizing Investment in India",
        "url": "https://en.wikipedia.org/wiki/Groww",
        "text": """Groww is an Indian investment platform founded by Lalit Keshre, Harsh Jain, Neeraj Singh, and Ishan Bansal in 2016. All four founders were former Flipkart employees. Groww started as a mutual fund distribution platform and expanded into stock trading, US stock investing, fixed deposits, and digital gold. The company has raised over $400 million in funding and was valued at $3 billion in 2022. Groww has over 10 million active investors on its platform, making it one of India's largest investment platforms. The app is known for its simple, user-friendly interface that makes investing accessible to first-time investors. In 2023, Groww acquired Indiabulls' mutual fund business for Rs 175 crore, obtaining its own Asset Management Company (AMC) license. Groww competes with Zerodha, Upstox, and Angel One in the discount broking space. The company has focused on financial literacy and education, offering free courses and content about investing. Groww's revenue crossed Rs 2,000 crore in FY2024 and the company achieved profitability.""",
        "authors": ["Wikipedia"],
        "publish_date": "2024-03-15"
    },
    {
        "id": "byju_010",
        "title": "BYJU'S: Rise and Crisis of India's EdTech Giant",
        "url": "https://en.wikipedia.org/wiki/Byju%27s",
        "text": """BYJU'S (Think and Learn Pvt Ltd) was founded by Byju Raveendran in 2011 and became India's most valuable startup at its peak valuation of $22 billion in 2022. The edtech company offered online learning programs for K-12 students and competitive exam preparation. BYJU'S raised over $5.5 billion from investors including Blackstone, Tiger Global, General Atlantic, and Chan Zuckerberg Initiative. However, starting in 2022, BYJU'S faced severe financial difficulties. The company undertook aggressive acquisitions including Aakash Educational Services ($950 million), WhiteHat Jr ($300 million), and Great Learning ($600 million), accumulating significant debt. By 2023, BYJU'S was embroiled in multiple controversies including delayed financial audits, mass layoffs of over 5,000 employees, investor lawsuits, and allegations of financial mismanagement. The company's valuation was marked down by several investors to near zero. In 2024, BYJU'S faced insolvency proceedings and the National Company Law Tribunal (NCLT) ordered the company's corporate insolvency resolution process. The BYJU'S saga serves as a cautionary tale about hyper-growth, excessive spending, and governance failures in the Indian startup ecosystem.""",
        "authors": ["Wikipedia"],
        "publish_date": "2024-01-30"
    },
    {
        "id": "paytm_011",
        "title": "Paytm: From Mobile Wallets to Financial Services Challenges",
        "url": "https://en.wikipedia.org/wiki/Paytm",
        "text": """Paytm (One97 Communications) was founded by Vijay Shekhar Sharma in 2010. Initially a mobile recharge platform, Paytm became synonymous with digital payments in India, especially after demonetization in 2016 when digital wallets saw massive adoption. The company raised over $4.7 billion from investors including SoftBank, Ant Financial (Alibaba), and Berkshire Hathaway. Paytm went public in November 2021 with India's largest IPO at the time, raising Rs 18,300 crore at a valuation of $20 billion. However, the stock fell 27% on listing day and continued to decline. In January 2024, the Reserve Bank of India (RBI) ordered Paytm Payments Bank to stop accepting new deposits and credit transactions, citing persistent compliance issues. This regulatory action severely impacted Paytm's business, forcing the company to migrate users to other banks. The Paytm Payments Bank crisis highlighted the importance of regulatory compliance in fintech. Despite these challenges, Paytm's core payment gateway business remained operational, processing transactions for millions of merchants. Vijay Shekhar Sharma stepped down as chairman of Paytm Payments Bank following the RBI order.""",
        "authors": ["Wikipedia"],
        "publish_date": "2024-02-25"
    },
    {
        "id": "startup_india_012",
        "title": "Startup India: Government Initiatives and Policy Impact",
        "url": "https://www.startupindia.gov.in",
        "text": """Startup India is a flagship initiative launched by the Government of India on January 16, 2016, to build a strong ecosystem for nurturing innovation and startups. The initiative aims to drive sustainable economic growth and generate large-scale employment opportunities. Key features of Startup India include tax exemptions for eligible startups for three consecutive years, simplified compliance procedures, and the Fund of Funds for Startups (FFS) with a corpus of Rs 10,000 crore managed through SIDBI. The DPIIT recognition program has recognized over 125,000 startups across India by 2024. India has become the world's third-largest startup ecosystem after the US and China, with over 100 unicorns. The government has also introduced the Startup India Seed Fund Scheme with Rs 945 crore to provide financial assistance to startups at the proof of concept stage. States like Karnataka, Maharashtra, and Telangana have emerged as major startup hubs. India produced 25 new unicorns in 2024, bringing the total to over 115. The ecosystem generated over 1.5 million direct jobs and 4 million indirect jobs. Key sectors include fintech, healthtech, edtech, SaaS, e-commerce, and deep tech.""",
        "authors": ["DPIIT"],
        "publish_date": "2024-03-20"
    },
    {
        "id": "upi_013",
        "title": "UPI Revolution: How India Built the World's Largest Real-Time Payment System",
        "url": "https://en.wikipedia.org/wiki/Unified_Payments_Interface",
        "text": """The Unified Payments Interface (UPI) is a real-time payment system developed by the National Payments Corporation of India (NPCI) and launched in April 2016. UPI enables instant money transfers between bank accounts through mobile devices. By 2024, UPI processed over 12 billion transactions monthly, making India the global leader in real-time digital payments. The total transaction value exceeded $2 trillion annually. UPI's success is attributed to its interoperability (works across all banks), zero transaction cost for consumers, and simple user experience (payments via phone number or QR code). Major UPI apps include PhonePe (48% market share), Google Pay (36%), and Paytm (12%). India's UPI model has attracted international attention, with countries like Singapore, UAE, Sri Lanka, and France implementing UPI-based payment systems. UPI has transformed India's economy by enabling digital payments for street vendors, auto-rickshaw drivers, and small shops. The NPCI introduced UPI Lite for small transactions under Rs 500 without requiring a PIN, and UPI 123PAY for feature phone users. UPI's success story demonstrates how public digital infrastructure can drive financial inclusion at massive scale.""",
        "authors": ["NPCI"],
        "publish_date": "2024-01-10"
    },
    {
        "id": "indian_saas_014",
        "title": "India's SaaS Revolution: Building for the World",
        "url": "https://en.wikipedia.org/wiki/Zoho",
        "text": """India has emerged as a global SaaS (Software as a Service) powerhouse, with Indian SaaS companies generating over $15 billion in annual revenue by 2024. The ecosystem is projected to reach $50 billion by 2030. Key Indian SaaS companies include Zoho (founded by Sridhar Vembu, bootstrapped to over $1 billion in revenue), Freshworks (founded by Girish Mathrubootham, IPO on NASDAQ at $12 billion valuation), Postman (API development platform valued at $5.6 billion), and Chargebee (subscription billing, valued at $3.5 billion). India produces more SaaS unicorns than any country except the US. The Indian SaaS model typically involves building products in India for global markets, leveraging India's engineering talent and cost advantages. Companies like Zoho have championed building from Tier 2 cities like Chennai, proving that world-class software can be built anywhere. Freshworks, founded in Chennai, became the first Indian SaaS company to list on NASDAQ in 2021. The SaaS ecosystem benefits from India's large pool of English-speaking engineers, competitive salaries, and growing domestic market. Key SaaS hubs include Chennai, Bangalore, Pune, and Hyderabad.""",
        "authors": ["Industry Report"],
        "publish_date": "2024-02-05"
    },
    {
        "id": "ai_india_015",
        "title": "AI Startups in India: The New Frontier",
        "url": "https://en.wikipedia.org/wiki/Artificial_intelligence_in_India",
        "text": """India's AI startup ecosystem has grown rapidly with over 3,000 AI startups by 2024. Key players include Krutrim AI (founded by Bhavish Aggarwal, India's first AI unicorn at $1 billion valuation), Sarvam AI (building foundation models for Indian languages, raised $41 million), and Fractal Analytics (AI solutions for enterprises, valued at $1.5 billion). The Indian government launched the IndiaAI Mission with a budget of Rs 10,372 crore to develop AI infrastructure including computing capacity of 10,000 GPUs. Indian AI startups are focusing on unique opportunities including multilingual AI models supporting 22 official Indian languages, AI for agriculture (serving 150 million farmers), healthcare AI for rural areas, and AI-powered financial inclusion. Companies like Niki.ai, Yellow.ai, and Haptik have built conversational AI platforms serving millions of users. India's AI talent pool is one of the largest globally, with over 400,000 AI professionals. IITs and IISc have established dedicated AI research centers. Key challenges include computing infrastructure costs, data quality and availability, and talent retention as global tech companies recruit aggressively from India. The Generative AI wave has spawned hundreds of new startups building applications on top of foundation models.""",
        "authors": ["Industry Report"],
        "publish_date": "2024-03-25"
    },
    {
        "id": "swiggy_016",
        "title": "Swiggy: India's Food Delivery and Quick Commerce Leader",
        "url": "https://en.wikipedia.org/wiki/Swiggy",
        "text": """Swiggy is an Indian food delivery and quick commerce platform founded by Sriharsha Majety, Nandan Reddy, and Rahul Jaimini in 2014. Headquartered in Bangalore, Swiggy pioneered the aggregator model for food delivery in India. The company has raised over $3.6 billion in funding from investors including SoftBank, Prosus, Accel, and DST Global. Swiggy went public in November 2024 with an IPO that valued the company at approximately $11.3 billion. The IPO was oversubscribed 3.6 times. Swiggy operates in three main segments: food delivery, Instamart (quick commerce delivering groceries in 10-15 minutes), and Swiggy Genie (pick-up and drop service). Instamart has become a major growth driver, competing with Blinkit (owned by Zomato) and Zepto in the quick commerce space. Swiggy partners with over 200,000 restaurants across 580+ cities and has a fleet of over 300,000 delivery partners. The company has invested heavily in its supply chain and dark store network for quick commerce. Swiggy's competition with Zomato defines India's food-tech landscape, with both companies expanding into adjacent services.""",
        "authors": ["Wikipedia"],
        "publish_date": "2024-11-15"
    },
    {
        "id": "zomato_017",
        "title": "Zomato: From Restaurant Discovery to Quick Commerce Dominance",
        "url": "https://en.wikipedia.org/wiki/Zomato",
        "text": """Zomato is an Indian food delivery and restaurant discovery platform founded by Deepinder Goyal and Pankaj Chaddah in 2008, originally as Foodiebay. The company rebranded to Zomato in 2010 and expanded to 24 countries before consolidating operations to focus on India. Zomato went public in July 2021 with a blockbuster IPO that valued the company at $13.1 billion. The stock initially surged but faced volatility. However, by 2024, Zomato's market cap exceeded $25 billion as the company turned profitable. Zomato's acquisition of Blinkit (formerly Grofers) for $568 million in 2022 proved transformative. Blinkit became India's leading quick commerce platform, delivering groceries and essentials in 10 minutes through a network of dark stores. Zomato reported revenues of Rs 14,000 crore in FY2024 with net profit crossing Rs 1,200 crore. The company introduced Zomato Gold, a subscription service, and expanded Hyperpure, its B2B ingredient supply business for restaurants. Deepinder Goyal's leadership through losses to profitability is studied as a case in sustainable growth. Zomato was included in the BSE Sensex index in 2024, becoming the youngest company to join the benchmark index.""",
        "authors": ["Wikipedia"],
        "publish_date": "2024-12-01"
    },
    {
        "id": "funding_winter_018",
        "title": "Indian Startup Funding Winter and Recovery: 2022-2025",
        "url": "https://analysis.startup-ecosystem.com",
        "text": """The Indian startup ecosystem experienced a significant funding downturn starting in late 2022, often called the 'funding winter.' Total startup funding dropped from $42 billion in 2021 to $24 billion in 2022 and further to $14 billion in 2023. The downturn was driven by rising global interest rates, the collapse of Silicon Valley Bank, and a correction in startup valuations after the ZIRP (Zero Interest Rate Policy) era. The funding winter led to widespread layoffs, with over 30,000 startup employees losing jobs in 2023 alone. Companies like BYJU'S, Ola, Swiggy, and ShareChat conducted major layoffs. The downturn forced startups to focus on profitability over growth, a fundamental shift in mindset. Many startups that had raised at inflated valuations faced down rounds or struggled to raise follow-on funding. However, 2024 showed signs of recovery with funding reaching $18 billion and 25 new unicorns emerging. Late-stage deals returned with major rounds for companies like PhonePe, Zepto, and Lenskart. The funding winter taught the ecosystem valuable lessons about sustainable growth, unit economics, and the importance of profitability. By 2025, Indian startup funding is projected to recover further with AI, climate tech, and deep tech leading investment interest.""",
        "authors": ["Startup Ecosystem Analysis"],
        "publish_date": "2024-12-15"
    },
    {
        "id": "delhivery_019",
        "title": "Delhivery: Building India's Logistics Infrastructure",
        "url": "https://en.wikipedia.org/wiki/Delhivery",
        "text": """Delhivery is an Indian logistics and supply chain company founded by Sahil Barua, Mohit Tandon, Bhavesh Manglani, Suraj Saharan, and Kapil Bharati in 2011. Starting as a hyperlocal delivery service in Gurgaon, Delhivery grew to become India's largest fully-integrated logistics services provider. The company went public in May 2022, raising Rs 5,235 crore in its IPO. Delhivery provides express parcel delivery, freight, warehousing, and cross-border logistics services. The company operates 24 automated sort centers, 115 gateways, 3,200+ direct delivery centers, and over 9,000 pin codes across India. Delhivery processes over 2 million shipments daily and has delivered over 2.5 billion orders cumulatively. The company serves major e-commerce platforms including Flipkart, Amazon, and Meesho, as well as D2C brands. Delhivery's technology platform uses machine learning for route optimization, demand forecasting, and real-time tracking. The company acquired Spoton Logistics in 2022 for Rs 3,325 crore, strengthening its B2B logistics capabilities. Despite initial post-IPO losses, Delhivery turned EBITDA positive in FY2024 and is focused on achieving full profitability.""",
        "authors": ["Wikipedia"],
        "publish_date": "2024-02-28"
    },
    {
        "id": "zepto_020",
        "title": "Zepto: The Quick Commerce Disruptor Founded by Teenagers",
        "url": "https://en.wikipedia.org/wiki/Zepto_(company)",
        "text": """Zepto is an Indian quick commerce startup founded by Aadit Palicha and Kaivalya Vohra in 2021, when both were just 19 years old and Stanford University dropouts. Zepto promises delivery of groceries and essentials in 10 minutes through a network of dark stores (micro-fulfillment centers). The company raised over $1.4 billion in funding and was valued at $5 billion in 2024. Zepto operates over 500 dark stores across major Indian cities. The company has grown from processing a few thousand orders per day to over 1 million daily orders by 2024. Zepto's rapid growth has been fueled by India's quick commerce boom, where consumers increasingly expect ultra-fast delivery. The company competes with Blinkit (Zomato) and Instamart (Swiggy) in the hyper-competitive quick commerce space. Zepto's technology stack uses real-time inventory management, demand prediction, and route optimization to achieve its 10-minute delivery promise. Aadit Palicha, at 22, became one of the youngest billionaires in India. Zepto's story exemplifies the new generation of Indian founders who are building category-defining companies at unprecedented speed. The company has been preparing for an IPO expected in 2025.""",
        "authors": ["Wikipedia"],
        "publish_date": "2024-09-10"
    }
]


def main():
    print("=" * 60)
    print("Seeding Indian startup knowledge base...")
    print("=" * 60)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for article in ARTICLES:
        path = RAW_DATA_DIR / f"{article['id']}.json"
        path.write_text(json.dumps(article, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  + {article['title'][:60]}")

    print(f"\nDone! Saved {len(ARTICLES)} articles to {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
