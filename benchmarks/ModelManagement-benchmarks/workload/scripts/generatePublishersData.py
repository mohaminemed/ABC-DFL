import json
from web3 import Web3

def generate_random_address():
    """Generate a random Ethereum address using Web3."""
    w3 = Web3()
    account = w3.eth.account.create()
    return account.address

def generate_addresses(n):
    """Generate a list of n Ethereum addresses."""
    return [{"publisher": generate_random_address()} for _ in range(n)]

def main():
    n = 10000  # Number of addresses to generate
    addresses = generate_addresses(n)
    
    # Write to JSON file
    with open('benchmarks/ModelManagement-benchmarks/workload/scripts/publishersData.json', 'w') as json_file:
        json.dump(addresses, json_file, indent=4)

if __name__ == "__main__":
    main()
