require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

module.exports = {
  solidity: "0.8.0",  // or the version you're using
  networks: {
    hardhat: {
      chainId: 1337,
    },
    besu: {
      url: "http://127.0.0.1:8545", // Besu RPC URL
      accounts: [process.env.SECRET_KEY], // Load private key from .env
    },
  },
};
