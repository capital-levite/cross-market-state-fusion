#!/usr/bin/env python3
"""
Data recorder for Cross-Market State Fusion.
Captures live MarketState sequences from Binance and Polymarket and saves them to disk.
"""
import asyncio
import json
import os
import time
import pickle
from datetime import datetime, timezone
from typing import Dict, List

from helpers import get_15m_markets, BinanceStreamer, OrderbookStreamer, FuturesStreamer, get_logger
from strategies.base import MarketState

class DataRecorder:
    def __init__(self, output_dir: str = "data/recordings"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Streamers
        self.price_streamer = BinanceStreamer(["BTC", "ETH", "SOL", "XRP"])
        self.orderbook_streamer = OrderbookStreamer()
        self.futures_streamer = FuturesStreamer(["BTC", "ETH", "SOL", "XRP"])
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(self.output_dir, f"market_data_{self.session_id}.pkl")
        
        self.running = False
        self.data_buffer = []
        self.markets = {}
        self.open_prices = {}

    async def refresh_markets(self):
        """Find active 15-min markets."""
        markets = get_15m_markets(assets=["BTC", "ETH", "SOL", "XRP"])
        now = datetime.now(timezone.utc)
        
        self.markets.clear()
        for m in markets:
            mins_left = (m.end_time - now).total_seconds() / 60
            if mins_left < 0.5: continue
            
            self.markets[m.condition_id] = m
            self.orderbook_streamer.subscribe(m.condition_id, m.token_up, m.token_down)
            
            current_price = self.price_streamer.get_price(m.asset)
            if current_price > 0:
                self.open_prices[m.condition_id] = current_price

    async def record_loop(self):
        tick = 0
        while self.running:
            await asyncio.sleep(1.0) # 1Hz recording
            now = datetime.now(timezone.utc)
            
            # Refresh markets every 5 minutes
            if tick % 300 == 0:
                await self.refresh_markets()
            
            tick += 1
            
            snapshot = {
                "timestamp": now.timestamp(),
                "states": {}
            }
            
            for cid, m in self.markets.items():
                if m.end_time <= now: continue
                
                # Build state (similar to run.py logic)
                state = MarketState(
                    asset=m.asset,
                    prob=m.price_up,
                    time_remaining=(m.end_time - now).total_seconds() / 900
                )
                
                # Orderbook
                ob = self.orderbook_streamer.get_orderbook(cid, "UP")
                if ob and ob.mid_price:
                    state.prob = ob.mid_price
                    state.best_bid = ob.best_bid or 0.0
                    state.best_ask = ob.best_ask or 0.0
                    state.spread = ob.spread or 0.0
                    
                    if ob.bids and ob.asks:
                        bid_vol_l1 = ob.bids[0][1] if ob.bids else 0
                        ask_vol_l1 = ob.asks[0][1] if ob.asks else 0
                        total_l1 = bid_vol_l1 + ask_vol_l1
                        state.order_book_imbalance_l1 = (bid_vol_l1 - ask_vol_l1) / total_l1 if total_l1 > 0 else 0.0

                # Binance
                binance_price = self.price_streamer.get_price(m.asset)
                state.binance_price = binance_price
                open_price = self.open_prices.get(cid, binance_price)
                if open_price > 0:
                    state.binance_change = (binance_price - open_price) / open_price
                
                # Futures
                futures = self.futures_streamer.get_state(m.asset)
                if futures:
                    state.cvd = futures.cvd
                    state.trade_flow_imbalance = futures.trade_flow_imbalance
                    state.returns_1m = futures.returns_1m
                    state.returns_5m = futures.returns_5m
                    state.returns_10m = futures.returns_10m
                    state.trade_intensity = futures.trade_intensity
                    state.large_trade_flag = futures.large_trade_flag
                
                snapshot["states"][cid] = state

            self.data_buffer.append(snapshot)
            
            # Periodically save to disk
            if len(self.data_buffer) >= 60: # Every minute
                self.flush()
                print(f"[{now.strftime('%H:%M:%S')}] Saved {len(self.data_buffer)} snapshots. Total time: {tick}s")

    def flush(self):
        if not self.data_buffer: return
        
        mode = 'ab' if os.path.exists(self.output_file) else 'wb'
        with open(self.output_file, mode) as f:
            pickle.dump(self.data_buffer, f)
        self.data_buffer = []

    async def run(self, duration_hours: float = 4.0):
        self.running = True
        print(f"Starting recorder: {self.output_file}")
        print(f"Target duration: {duration_hours} hours")
        
        await self.refresh_markets()
        
        tasks = [
            self.price_streamer.stream(),
            self.orderbook_streamer.stream(),
            self.futures_streamer.stream(),
            self.record_loop()
        ]
        
        try:
            await asyncio.wait(tasks, timeout=duration_hours * 3600)
        except asyncio.TimeoutError:
            print("Recording duration reached.")
        finally:
            self.running = False
            self.flush()
            print("Recorder stopped.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=4.0)
    args = parser.parse_args()
    
    recorder = DataRecorder()
    asyncio.run(recorder.run(duration_hours=args.hours))
