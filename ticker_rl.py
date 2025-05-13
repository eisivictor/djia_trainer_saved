from claude_rl_infra import get_historical_data, DQNAgent, StockTradingEnv, train_agent, testing_agent
import argparse
import os
import logging

def train_model(logger, ticker, model_filename, lookback, gamma, batch_size, learning_rate, epsilon_initial, 
               epsilon_final, epsilon_decay, memory_size, episodes, initial_capital, period):
    """
    Train the DQN trading model and save the weights
    
    Parameters:
    - ticker: Stock ticker symbol
    - lookback: Lookback window size
    - gamma: Discount factor
    - batch_size: Batch size for training
    - learning_rate: Learning rate
    - epsilon_initial: Initial exploration rate
    - epsilon_final: Final exploration rate
    - epsilon_decay: Epsilon decay steps
    - memory_size: Replay memory size
    - episodes: Training episodes
    - initial_capital: Initial capital
    - period: Data period in years
    
    Returns:
    - agent: Trained DQN agent
    - data: Historical data used for training
    """
    # Get stock data
    logger.info(f"Fetching {ticker} historical data...")
    data = get_historical_data(ticker, period=period)
    
    # Create training environment
    train_env = StockTradingEnv(data, initial_balance=initial_capital,
                               lookback_window_size=lookback)
    
    # Define state and action sizes
    state_size = train_env.observation_space.shape
    action_size = train_env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_size, action_size, epsilon_initial,
                     memory_size, epsilon_final, epsilon_decay, gamma, learning_rate)
    
    # Load the trained model if exists
    if os.path.exists(model_filename):
        logger.info(f"Loading existing model weights from {model_filename}...")
        agent.load(model_filename)
    
    # Train agent
    logger.info("\nTraining the agent...")
    train_scores = train_agent(logger, train_env, agent, episodes=episodes, batch_size=batch_size)
    
    # Save the trained model
    logger.info(f"Saving model weights to {model_filename}...")
    agent.save(model_filename)
    
    return agent, data

def test_model(logger, ticker, lookback, initial_capital, period, model_weights_path=None, agent=None, data=None):
    """
    Test the trained DQN model on historical data
    
    Parameters:
    - ticker: Stock ticker symbol
    - lookback: Lookback window size
    - initial_capital: Initial capital
    - period: Data period in years
    - model_weights_path: Path to the model weights file
    - agent: Optional pre-trained agent (if None, will load from saved weights)
    - data: Optional historical data (if None, will fetch new data)
    
    Returns:
    - test_results: Results from testing the agent
    """
    # If no agent is provided, create one and load weights
    if agent is None:
        # Get stock data if not provided
        if data is None:
            logger.info(f"Fetching {ticker} historical data for testing...")
            data = get_historical_data(ticker, period)

        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        
        # Create testing environment to get state and action sizes
        test_env = StockTradingEnv(data, initial_balance=initial_capital,
                                  lookback_window_size=lookback)
        
        # Define state and action sizes
        state_size = test_env.observation_space.shape
        action_size = test_env.action_space.n
        
        # Create agent
        agent = DQNAgent(state_size, action_size, 
                         epsilon_initial=0.01, memory_size=10000, epsilon_final=0.01, epsilon_decay_steps=1, gamma=0.99, learning_rate=0.001)
                
        # Check if model weights path is provided and exists
        if not model_weights_path or not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Model file {model_weights_path} not found. Train the model first.")
        
        logger.info(f"Loading model weights from {model_weights_path} for testing...")
        agent.load(model_weights_path)
    
    # Test agent
    logger.info("\nTesting the agent...")
    test_results = testing_agent(logger, ticker.lower(), agent, data, lookback, initial_capital)
    
    # Extract transactions from the test results
    transactions = test_results.get('transactions', [])
        
    return {
        "results": test_results,
        "transactions": transactions  # Include transactions in the returned data
    }

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Stock Trading Agent with DQN')
    
    # Add arguments
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--lookback', type=int, default=20, help='Lookback window size (default: 20)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (default: 0.99)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--epsilon_initial', type=float, default=1.0, help='Initial exploration rate (default: 1.0)')
    parser.add_argument('--epsilon_final', type=float, default=0.01, help='Final exploration rate (default: 0.01)')
    parser.add_argument('--epsilon_decay', type=int, default=10000, help='Epsilon decay steps (default: 10000)')
    parser.add_argument('--memory_size', type=int, default=10000, help='Replay memory size (default: 10000)')
    parser.add_argument('--episodes', type=int, default=40, help='Training episodes (default: 40)')
    parser.add_argument('--initial_capital', type=float, default=10000, help='Initial capital (default: 10000)')
    parser.add_argument('--period', type=int, default=1, help='Data period in years (default: 1)')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both', 
                        help='Operation mode: train, test, or both (default: both)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging to stdout
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    
    if args.mode == 'train' or args.mode == 'both':
        # Train the model
        agent, data = train_model(
            logger, args.ticker, args.lookback, args.gamma, args.batch_size, args.learning_rate,
            args.epsilon_initial, args.epsilon_final, args.epsilon_decay,
            args.memory_size, args.episodes, args.initial_capital, args.period
        )
        
        # If both modes, use the trained agent and data for testing
        if args.mode == 'both':
            test_model(logger, args.ticker, args.lookback, args.initial_capital, args.period, agent=agent, data=data)
    
    elif args.mode == 'test':
        # Only test the model using saved weights
        model_filename = f"{args.ticker.lower()}_trading_model.keras"
        test_model(logger, args.ticker, args.lookback, args.initial_capital, args.period, model_weights_path=model_filename)

if __name__ == "__main__":
    main()