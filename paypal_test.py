import paypalrestsdk
import webbrowser
import hashlib
import json
import time
import threading
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
import heapq
import random
import statistics
from cryptography.fernet import Fernet
import sqlite3
import logging

# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# singleton pattern for configuration management
class payPalConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(payPalConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.config = {
            "mode": "sandbox",
            "client_id": "AVu7TGGK1XdJUSyV11J4Vt0HC4zxZDPj_5rHydHmny6EQgRq3tzlYkPhzTiLfFmTBOwTJIkMNuufJcZw",
            "client_secret": "EOULt0fiBsfwF4gUJjcyxKhqqDKCe5lTvQgTb2KE8pC5UXi7GY-Cho_t_lj0IP-NqkVCaSDku5hv4q8C"
        }
        paypalrestsdk.configure(self.config)

    def get_config(self):
        return self.config


# Factory pattern for payment creation
class PaymentFactory:
    @staticmethod
    def create_payment(intent: str, amount: float, currency: str, description: str, return_url: str, cancel_url: str) -> paypalrestsdk.Payment:
        payment_info = {
            "intent": intent,
            "payer": {
                "payment_method": "paypal"
            },
            "redirect_urls": {
                "return_url": return_url,
                "cancel_url": cancel_url
            },
            "transactions": [{
                "amount": {
                    "total": f"{amount:.2f}",
                    "currency": currency
                },
                "description": description
            }]
        }

        return paypalrestsdk.Payment(payment_info)


# observer pattern for payment status updates
class PaymentObserver:
    def update(self, payment_id: str, status: str):
        pass


class EmailNotifier(PaymentObserver):
    def update(self, payment_id: str, status: str):
        logger.info(f"Email sent: payment {payment_id} status changed to {status}")


class SMSnotifier(PaymentObserver):
    def update(self, payment_id: str, status: str):
        logger.info(f"SMS sent: payment {payment_id} status changed to {status}")


# strategy pattern for payment validation
class PaymentValidation:
    def validate(self, payment: paypalrestsdk.Payment) -> bool:
        pass


class AmountvalidationStrategy(PaymentValidation):
    def validate(self, payment: paypalrestsdk.Payment) -> bool:
        try:
            amount = float(payment.transactions[0].amount.total)
            return amount > 0
        except (ValueError, IndexError, AttributeError):
            return False


class CurrencyValidationStrategy(PaymentValidation):
    def __init__(self, allowed_currencies: List[str] = None):
        self.allowed_currencies = allowed_currencies or ["USD", "EUR", "GBP"]

    def validate(self, payment: paypalrestsdk.Payment) -> bool:
        try:
            currency = payment.transactions[0].amount.currency
            return currency in self.allowed_currencies
        except (IndexError, AttributeError):
            return False


# Composite validator using multiple strategies
class CompositePaymentValidator(PaymentValidation):
    def __init__(self, strategies: List[PaymentValidation]):
        self.strategies = strategies

    def validate(self, payment: paypalrestsdk.Payment) -> bool:
        return all(strategy.validate(payment) for strategy in self.strategies)


# Rate limiting using token bucket algorithm
class RateLimiter:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def allow_request(self) -> bool:
        with self.lock:
            self.refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        if new_tokens > 0:
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now


# circuit breaker pattern for handling paypal API failures
class circuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = "CLOSED"
        self.last_failure_time = 0
        self.lock = threading.Lock()

    def execute(self, func, *args, **kwargs):
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF-OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            with self.lock:
                if self.state == "HALF-OPEN":
                    self.state = "CLOSED"
                    self.failures = 0
            return result
        except Exception as e:
            self._record_failure()
            raise e

    def _record_failure(self):
        with self.lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"


# Database layer for persistence using sqlite
class PaymentDatabase:
    def __init__(self, db_path: str = "payments.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS payments (
                           id TEXT PRIMARY KEY,
                           intent TEXT,
                           amount REAL,
                           currency TEXT,
                           description TEXT,
                           status TEXT,
                           created_at TIMESTAMP,
                           updated_at TIMESTAMP
                           )
                ''')
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS payment_events(
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           payment_id TEXT,
                           event_type TEXT,
                           event_data TEXT,
                           created_at TIMESTAMP,
                           FOREIGN KEY (payment_id) REFERENCES payments (id)
                           )
                           ''')
            conn.commit()

    def save_payment(self, payment: paypalrestsdk.Payment):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                           INSERT OR REPLACE INTO payments
                           (id,intent,amount,currency,description,status,created_at,updated_at)
                           VALUES (?,?,?,?,?,?,?,?)
                           ''', (
                payment.id,
                payment.intent,
                float(payment.transactions[0].amount.total),
                payment.transactions[0].amount.currency,
                payment.transactions[0].description,
                payment.state,
                datetime.now(),
                datetime.now()
            ))
            conn.commit()

    def record_event(self, payment_id: str, event_type: str, event_data: dict):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                           INSERT INTO payment_events (payment_id,event_type,event_data,created_at)
                           VALUES (?,?,?,?)
                           ''', (payment_id, event_type, json.dumps(event_data), datetime.now())
                           )
            conn.commit()


# cache for frequently accessed payment data using LRU algorithm
class PaymentCache:
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.cache = {}
        self.access_order = deque()
        self.lock = threading.Lock()

    def get(self, payment_id: str) -> Optional[dict]:
        with self.lock:
            if payment_id in self.cache:
                # move to end to mark as recently used
                try:
                    self.access_order.remove(payment_id)
                except ValueError:
                    # already not present in access_order for some reason
                    pass
                self.access_order.append(payment_id)
                return self.cache[payment_id]
        return None

    def put(self, payment_id: str, payment_data: dict):
        with self.lock:
            if payment_id in self.cache:
                # update the existing entry
                try:
                    self.access_order.remove(payment_id)
                except ValueError:
                    pass
            elif len(self.cache) >= self.capacity:
                # remove the least recently used
                lru = self.access_order.popleft()
                del self.cache[lru]

            self.cache[payment_id] = payment_data
            self.access_order.append(payment_id)


# priority queue for processing payments based on the amount (larger amounts get higher priority)
class PriorityPaymentQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0  # handle same priority payments

    def push(self, payment: paypalrestsdk.Payment):
        # use negative amount for max-heap behaviour (larger amounts have a higher priority)
        amount = float(payment.transactions[0].amount.total)
        heapq.heappush(self.heap, (-amount, self.counter, payment))
        self.counter += 1

    def pop(self) -> Optional[paypalrestsdk.Payment]:
        if self.heap:
            _, _, payment = heapq.heappop(self.heap)
            return payment
        return None

    def is_empty(self) -> bool:
        return len(self.heap) == 0


# Exponential backoff for retrying failed requests
def exponential_backoff(retries: int, base_delay: float = 1.0, max_delay: float = 60.0):
    delay = min(max_delay, base_delay * (2 ** (retries - 1)))
    jitter = random.uniform(0, delay * 0.1)  # add jitter to avoid thundering herd
    time.sleep(delay + jitter)


# payment service with all enhanced features
class PaymentService:
    def __init__(self):
        self.config = payPalConfig()
        self.observers = []
        self.validator = CompositePaymentValidator([
            AmountvalidationStrategy(),
            CurrencyValidationStrategy()
        ])

        self.rate_limiter = RateLimiter(capacity=100, refill_rate=10)  # 100 requests , refill 10 per second
        self.circuit_breaker = circuitBreaker(failure_threshold=5, recovery_timeout=60)

        self.db = PaymentDatabase()
        self.cache = PaymentCache(capacity=100)
        self.priority_queue = PriorityPaymentQueue()

    def add_observer(self, observer: PaymentObserver):
        self.observers.append(observer)

    def notify_observers(self, payment_id: str, status: str):
        for observer in self.observers:
            observer.update(payment_id, status)

    def create_payment(self, intent: str, amount: float, currency: str, description: str,
                       return_url: str, cancel_url: str, max_retries: int = 3) -> Optional[paypalrestsdk.Payment]:
        # Rate limiting
        if not self.rate_limiter.allow_request():
            logger.warning("rate limit exceeded")
            return None

        # create payment objectd
        payment = PaymentFactory.create_payment(
            intent, amount, currency, description, return_url, cancel_url
        )

        # validate payment
        if not self.validator.validate(payment):
            logger.error("payment validation failed")
            return None

        # add to priority queue for processing
        self.priority_queue.push(payment)
        retries = 0
        while retries <= max_retries:
            try:
                result = self.circuit_breaker.execute(self._create_payment_impl, payment)
                return result
            except Exception as e:
                retries += 1
                logger.error(f"payment processing failed attempt {retries}/{max_retries}: {str(e)}")
                if retries <= max_retries:
                    exponential_backoff(retries)
                else:
                    logger.error("Max retries exceeded")
                    return None

    def _create_payment_impl(self, payment: paypalrestsdk.Payment) -> paypalrestsdk.Payment:
        if payment.create():
            logger.info(f"Payment created successfully: {payment.id}")

            # save to database
            self.db.save_payment(payment)
            self.db.record_event(payment.id, "created", {"status": payment.state})

            # cache the payment data
            payment_data = {
                "id": payment.id,
                "intent": payment.intent,
                "amount": float(payment.transactions[0].amount.total),
                "currency": payment.transactions[0].amount.currency,
                "status": payment.state,
                "created_at": datetime.now().isoformat()
            }

            self.cache.put(payment.id, payment_data)

            # notify observers
            self.notify_observers(payment.id, payment.state)

            # approval url
            for link in payment.links:
                if getattr(link, "rel", None) == "approval_url":
                    return payment

            logger.error("approval URL not found")
            raise Exception("Approval URL not found")
        else:
            logger.error(f"Payment creation failed: {payment.error}")
            raise Exception(f"payment creation failed: {payment.error}")

    def get_payment_status(self, payment_id: str) -> Optional[str]:
        # check cache first
        cached = self.cache.get(payment_id)
        if cached:
            return cached.get("status")

        # if not in cache , query the database
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM payments WHERE id = ?", (payment_id,))
            result = cursor.fetchone()
            if result:
                return result[0]
        return None

    def process_queued_payments(self):
        while not self.priority_queue.is_empty():
            payment = self.priority_queue.pop()
            if payment:
                try:
                    self.create_payment(
                        payment.intent,
                        float(payment.transactions[0].amount.total),
                        payment.transactions[0].amount.currency,
                        payment.transactions[0].description,
                        payment.redirect_urls.get("return_url") if isinstance(payment.redirect_urls, dict) else getattr(payment.redirect_urls, "return_url", ""),
                        payment.redirect_urls.get("cancel_url") if isinstance(payment.redirect_urls, dict) else getattr(payment.redirect_urls, "cancel_url", "")
                    )
                except Exception as e:
                    logger.error(f"Failed to process queued payment: {str(e)}")


# Main Application
def main():
    # initialize
    payment_service = PaymentService()

    payment_service.add_observer(EmailNotifier())
    payment_service.add_observer(SMSnotifier())

    # create a test payment
    print("creating a test payment for $1.00...")

    payment = payment_service.create_payment(
        intent='sale',
        amount=1.00,
        currency="USD",
        description="this is test payment for my tutorial",
        return_url="http://example.com/success",
        cancel_url="http://example.com/cancel"
    )

    if payment:
        # find approval URL
        for link in payment.links:
            if getattr(link, "rel", None) == "approval_url":
                approval_url = getattr(link, "href", None) or link.href
                if approval_url:
                    print("Now opening paypal website for you to approve the payment...")
                    webbrowser.open(approval_url)
                    break
    else:
        print("Failed to create payment")


if __name__ == "__main__":
    main()


            
"""paypalrestsdk.configure({
    "mode":"sandbox",
    "client_id":"AW2FR-vpnmY5LE64hgfHPz4BxdaCcb0rdt61AHkpMAQAIewAvZKNWfmox407C3YXHxaccx_Js4DwKVR3",
    "client_secret":"EIuhSos_xBiNpgIkD5T3wkvLdwTrQ-WYi9JAX66LnTqbpFYTKPD00QyonMHYB3tgy0aU_zpcaZ50HB6v"

})

#create a payment
print("Creating a test payment for $1.00...")

#info we  send to paypal

payment_info={
    "intent":"sale",
    "payer":{
        "payment_method":"paypal"

    },
    "redirect_urls":{
        "return_url":"http://example.com/success",
        "cancel_url":"http://example.com/cancel"

    },

    "transactions":[{
        "amount":{
            "total":"1.00",
            "currency":"USD"
        },
        "description":"this is a test payment for my tutorial"

    }]
}

#create the payment on paypals servers

payment =paypalrestsdk.Payment(payment_info)

#try
if payment.create():
    print("Payment created successfully!")

    #link where user can approve the paymennt 

    for link in payment.links :
        if link.rel=="approval_url":
            approval_url=link.href

            print("now opening the paypal website for you to approve the payment...")

            webbrowser.open(approval_url)
            break
else:
    print("oh no  ! there was an error creating the payment:")
    print(payment.error)"""