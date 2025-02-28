const hre = require("hardhat");

async function main() {
  // Deploy AccessManagement Contract
  const AccessManagement = await hre.ethers.getContractFactory("AccessManagement");
  const accessManagement = await AccessManagement.deploy(); // Deploy the contract
  await accessManagement.waitForDeployment(); // Wait for deployment to complete
  console.log(`AccessManagement deployed to: ${await accessManagement.getAddress()}`);

  // Deploy ModelManagement Contract
  const ModelManagement = await hre.ethers.getContractFactory("ModelManagement");
  const modelManagement = await ModelManagement.deploy(); // Pass AccessManagement address if needed
  await modelManagement.waitForDeployment();
  console.log(`ModelManagement deployed to: ${await modelManagement.getAddress()}`);
}

// Run the deployment script
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
