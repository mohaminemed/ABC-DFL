const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ModelManagement", function () {
  let AccessManagement;
  let accessManagement;
  let ModelManagement;
  let modelManagement;
  let owner;
  let oracle1;
  let publisher;
  let chargingStation1;
  let chargingStation2;
  let publisherSigner;
  let csSigner1;
  let csSigner2;

  beforeEach(async function () {
    [owner, oracle1 , publisher, chargingStation1, chargingStation2] = await ethers.getSigners();
    publisherSigner = publisher;
    csSigner1 = chargingStation1;
    csSigner2 = chargingStation2;

    const AccessManagementFactory = await ethers.getContractFactory("AccessManagement");
    accessManagement = await AccessManagementFactory.deploy();

    const ModelManagementFactory = await ethers.getContractFactory("ModelManagement");
    modelManagement = await ModelManagementFactory.deploy();
  });

  describe("Model Publisher Registration", function () {
    it("should allow a model publisher to register", async function () {
      const tx = await modelManagement.connect(owner).registerModelPublisher(publisher.address);
      const receipt = await tx.wait();
      console.log(`Gas used for registerModelPublisher: ${receipt.gasUsed.toString()}`);
      
      await expect(tx)
        .to.emit(modelManagement, "ModelPublisherRegistered")
        .withArgs(publisher.address);
    });

    it("should not allow a model publisher to register twice", async function () {
      await modelManagement.connect(owner).registerModelPublisher(publisher.address);
      await expect(modelManagement.connect(owner).registerModelPublisher(publisher.address))
        .to.be.revertedWith("Model publisher already registered.");
    });
  });

  describe("Charging Station Registration", function () {
    it("should allow a charging station to register", async function () {
      const tx = await modelManagement.connect(owner).registerChargingStation(chargingStation1.address);
      const receipt = await tx.wait();
      console.log(`Gas used for registerChargingStation: ${receipt.gasUsed.toString()}`);
      
      await expect(tx)
        .to.emit(modelManagement, "ChargingStationRegistered")
        .withArgs(chargingStation1.address);
    });

    it("should not allow the same charging station to register twice", async function () {
      await modelManagement.connect(owner).registerChargingStation(chargingStation1.address);
      await expect(modelManagement.connect(owner).registerChargingStation(chargingStation1.address))
        .to.be.revertedWith("Charging station already registered.");
    });
  });

  describe("Model Publishing", function () {
    it("should allow a registered model publisher to publish a model", async function () {
      await modelManagement.connect(owner).registerModelPublisher(publisher.address);
      const tx = await modelManagement.connect(publisherSigner).publishModel("modelCID", "infoCID", 10, 3);
      const receipt = await tx.wait();
      console.log(`Gas used for publishModel: ${receipt.gasUsed.toString()}`);
      
      await expect(tx)
        .to.emit(modelManagement, "ModelPublished")
        .withArgs(0, "modelCID");
    });
  });

  describe("Charging Station Joining Model", function () {
    it("should allow a charging station to join a model", async function () {
        await modelManagement.connect(owner).registerModelPublisher(publisher.address);
        await modelManagement.connect(publisherSigner).publishModel("modelCID", "infoCID", 10, 3);
        await modelManagement.connect(owner).registerChargingStation(chargingStation1.address);

        // Assume deposit amount is 1 ETH 
        const depositAmount = ethers.parseEther("1.0"); 

        const tx = await modelManagement.connect(chargingStation1).joinChargingStationModel(0, { value: depositAmount });
        const receipt = await tx.wait();
        console.log(`Gas used for joinChargingStationModel: ${receipt.gasUsed.toString()}`);

        await expect(tx)
            .to.emit(modelManagement, "ChargingStationJoinedModel")
            .withArgs(0, chargingStation1.address, depositAmount);
    });
});


  describe("Intermediate Model Submission", function () {
    async function generateSignatures() {
      const { Wallet } = ethers;
      const EVs = [...Array(5)].map(() => Wallet.createRandom());
      const cid = "modelCID";
  
      const signatures = await Promise.all(
        EVs.slice(0, 3).map(async (ev) => ({
          signer: ev.address,
          signature: await signCID(ev, cid),
        }))
      );
  
      return signatures;
    }
  
    async function signCID(signer, cid) {
      const hash = ethers.solidityPackedKeccak256(["string"], [cid]);
      return await signer.signMessage(ethers.toBeArray(hash));
    }
  
    it("should allow a registered charging station to submit an intermediate model with valid signatures", async function () {
      await modelManagement.connect(owner).registerModelPublisher(publisher.address);
      await modelManagement.connect(publisherSigner).publishModel("modelCID", "infoCID", 10, 3);
      await modelManagement.connect(owner).registerChargingStation(chargingStation1.address);

      const depositAmount = ethers.parseEther("1.0"); 
      await modelManagement.connect(chargingStation1).joinChargingStationModel(0, { value: depositAmount });
  
      const signatures = await generateSignatures();
      const validSignatures = signatures.slice(0, 4);
      const signers = validSignatures.map(sig => sig.signer);
      const sigs = validSignatures.map(sig => sig.signature);
  
      const hash = ethers.solidityPackedKeccak256(["string"], ["modelCID"]);
      const tx = await modelManagement.connect(chargingStation1).submitIntermediateModel(0, hash, sigs, signers);
      const receipt = await tx.wait();
      console.log(`Gas used for submitIntermediateModel: ${receipt.gasUsed.toString()}`);
  
      const storedHash = await modelManagement.submittedHashes(0, 0, chargingStation1.address);
      expect(storedHash).to.equal(hash);
    });
  });

  describe("Global Model Submission", function () {
    async function generateSignatures(cid) {
      const { Wallet } = ethers;
      const oracles = [...Array(5)].map(() => Wallet.createRandom());
  
      const signatures = await Promise.all(
        oracles.slice(0, 3).map(async (oracle) => ({
          signer: oracle.address,
          signature: await signCID(oracle, cid),
        }))
      );
  
      return signatures;
    }
  
    async function signCID(signer, cid) {
      const hash = ethers.solidityPackedKeccak256(["string"], [cid]);
      return await signer.signMessage(ethers.toBeArray(hash));
    }
  
    it("should allow a registered oracle to submit a global model with valid signatures", async function () {
      await modelManagement.connect(owner).registerModelPublisher(publisher.address);
      await modelManagement.connect(publisherSigner).publishModel("modelCID", "infoCID", 10, 3);
      await modelManagement.connect(owner).registerOracle(oracle1.address);
  
      const cid = "globalModelCID";
      const signatures = await generateSignatures(cid);
      const validSignatures = signatures.slice(0, 4);
      const signers = validSignatures.map(sig => sig.signer);
      const sigs = validSignatures.map(sig => sig.signature);
  
      const hash = ethers.solidityPackedKeccak256(["string"], [cid]);
      const tx = await modelManagement.connect(oracle1).submitGlobalModel(0, hash, sigs, signers);
      const receipt = await tx.wait();
      console.log(`Gas used for submitGlobalModel: ${receipt.gasUsed.toString()}`);
  
      const storedHash = await modelManagement.globalModelHashes(0, 0);
      expect(storedHash).to.equal(hash);
    });
  
    it("should revert if a non-oracle attempts to submit a global model", async function () {
      await modelManagement.connect(owner).registerModelPublisher(publisher.address);
      await modelManagement.connect(publisherSigner).publishModel("modelCID", "infoCID", 10, 3);
  
      const cid = "invalidModelCID";
      const signatures = await generateSignatures(cid);
      const signers = signatures.map(sig => sig.signer);
      const sigs = signatures.map(sig => sig.signature);
      const hash = ethers.solidityPackedKeccak256(["string"], [cid]);
  
      await expect(
        modelManagement.connect(chargingStation1).submitGlobalModel(0, hash, sigs, signers)
      ).to.be.revertedWith("Only registered oracles can submit");
    });
  });

  const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Charging Station Selection and Reward Distribution", function () {
  
  async function signMessage(signer, _modelId, CSs) {
    const hash = ethers.solidityPackedKeccak256(["uint256", "address[]"], [_modelId, CSs]);
    return await signer.signMessage(ethers.toBeArray(hash));
  }

  async function generateSignatures(_modelId, CSs) {
    const { Wallet } = ethers;
    const oracles = [...Array(5)].map(() => Wallet.createRandom());

    const signatures = await Promise.all(
      oracles.slice(0, 3).map(async (oracle) => ({
        signer: oracle.address,
        signature: await signMessage(oracle,_modelId, CSs),
      }))
    );

    return signatures;
  }

  it("should allow a registered oracle to update CS selection count with valid signatures", async function () {
    await modelManagement.connect(owner).registerOracle(oracle1.address);
    await modelManagement.connect(owner).registerModelPublisher(publisher.address);
    await modelManagement.connect(publisherSigner).publishModel("modelCID", "infoCID", 10, 3);
    await modelManagement.connect(owner).registerChargingStation(chargingStation1.address);
    const depositAmount = ethers.parseEther("1.0"); 
    await modelManagement.connect(chargingStation1).joinChargingStationModel(0, { value: depositAmount });
 
    const benignCSs = [chargingStation1.address, chargingStation2.address];
    const signaturesData = await generateSignatures(0, benignCSs);

    const signersAddresses = signaturesData.map(sig => sig.signer);
    const signatures = signaturesData.map(sig => sig.signature);

    const tx =  await  modelManagement.connect(oracle1).updateCSSelectionCount(0, benignCSs, signatures, signersAddresses);
    const receipt = await tx.wait();
    console.log(`Gas used for updateCSSelectionCount: ${receipt.gasUsed.toString()}`);

    expect(await  modelManagement.csSelectionCount(chargingStation1.address)).to.equal(1);
    expect(await  modelManagement.csSelectionCount(chargingStation2.address)).to.equal(1);
  });

  it("should distribute rewards correctly based on CS selection counts", async function () {

    await modelManagement.connect(owner).registerOracle(oracle1.address);
    await modelManagement.connect(owner).registerModelPublisher(publisher.address);
    await modelManagement.connect(publisherSigner).publishModel("modelCID", "infoCID", 10, 3);
    await modelManagement.connect(owner).registerChargingStation(chargingStation1.address);
    const depositAmount = ethers.parseEther("1.0"); 
    await modelManagement.connect(chargingStation1).joinChargingStationModel(0, { value: depositAmount });

    
    const CSs = [chargingStation1.address, chargingStation2.address];
    const signaturesData = await generateSignatures(0, CSs);

    const signersAddresses = signaturesData.map(sig => sig.signer);
    const signatures = signaturesData.map(sig => sig.signature);

    // Simulate training rounds and deposits
    const rewardAmount = ethers.parseEther("10.0"); 
    await  modelManagement.connect(publisher).makeDepositToStartTraining(0, { value: rewardAmount });
    await  modelManagement.connect(oracle1).updateCSSelectionCount(0, [chargingStation1.address, chargingStation2.address], signatures, signersAddresses);

    const chargingStation1InitialBalance = await ethers.provider.getBalance(chargingStation1.address);
    const chargingStation2InitialBalance = await ethers.provider.getBalance(chargingStation2.address);

    const tx =  await  modelManagement.connect(oracle1).distributeReward(0, CSs, signatures, signersAddresses);
    const receipt = await tx.wait();
    console.log(`Gas used for distributeReward: ${receipt.gasUsed.toString()}`);

    expect(await ethers.provider.getBalance(chargingStation1.address)).to.be.above(chargingStation1InitialBalance);
    expect(await ethers.provider.getBalance(chargingStation2.address)).to.be.above(chargingStation2InitialBalance);
  });
});

  
});
