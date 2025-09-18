// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

import "./AccessManagement.sol";


contract ModelManagement is AccessManagement {

    struct Update {
        uint trainingAccuracy;
        uint trainingDataPoints;
        string weights; // CID
    }

    struct Score {
        address chargingStation;
        uint256 score;
    }

    struct ChargingStation {
        address _chargingStationAddr;
        //uint256 _reputation;
    }

    struct Model {
        uint modelId;
        string modelCID;
        string infoCID;
        address publisher;
        address[] chargingStations;
        string globalModelWeightsCID;
        uint currentRound;
        uint maxRounds;
        uint requiredChargingStations;
        address[] registeredChargingStations;
        string state;
    }

    uint256 _scaleRep = 1e18;
    uint public threshold = 2;  // Example threshold for signatures

    
    address[] public oracles;
    address[] public modelPublishers;
    address[] public chargingStations;
    mapping(address => bool) public registeredOracles;
    mapping(address => bool) public registeredChargingStations;
    mapping(address => bool) public registeredModelPublishers;

    mapping(uint => mapping(address => uint[])) public modelSelectedChargingStations;
    mapping(uint => mapping(uint => uint)) updatesCount;
    mapping(uint => mapping(uint => mapping(address => bool))) updatesSubmitted;
    mapping(uint => mapping(uint => mapping(address => Update))) public updates;
    //mapping(address => uint256) public publisherInteractions;
    mapping(address => mapping(address => uint256)) public interactionsChargingStationWithPublisher;
    mapping(uint => mapping(uint => Score[])) public scores;
    mapping(address => uint256) public balances;
    Model[] public models;

    // Stores model publisher's deposit (reward pool)
    mapping(uint => uint) public modelDeposits; 
    // Stores each charging station's deposit per model
    mapping(uint => mapping(address => uint)) public chargingStationDeposits; 
    mapping(uint256 => address[]) public modelChargingStations;
    mapping(address => uint256) public reputation; // Reputation scores for CSs
    mapping(address => uint256) public csSelectionCount; // How many times CS was in the largest cluster
    mapping(uint256 => uint256) public modelTrainingRounds; // Tracks total rounds per model
    mapping(uint => mapping(uint => mapping(address => bytes32))) public submittedHashes;
    mapping(uint => mapping(uint => bytes32)) public globalModelHashes;
    mapping(uint => mapping(uint => bool)) public globalModelSubmitted;
   

    
    event DepositReleased(uint256 indexed modelId, uint256 amountPerCS);
    event OracleRegistered(address indexed modelPublisher);
    event ModelPublisherRegistered(address indexed modelPublisher);
    event ChargingStationRegistered(address indexed chargingStation);
    event ModelPublished(uint indexed modelId, string CID);
    event TrainingDepositMade(uint modelId, address indexed publisher, uint amount);
    event ChargingStationJoinedModel(uint modelId, address indexed cs, uint amount);
    event IntermediateModelSubmitted(uint indexed modelId, address indexed cs, uint round, bytes32 hashCID);
    event GlobalModelSubmitted(uint indexed modelId, address indexed oracle, uint round, bytes32 hashGMCID);
    event CSSelectionUpdated(uint256 indexed modelId, address[] benignCSs);
    event RewardDistributed(uint256 indexed modelId, uint256 totalReward);
    event ReputationUpdated(address indexed cs, uint256 newReputation);


    constructor() {}

     function registerOracle(address _oracle) public onlyOwner {
        require(!registeredOracles[_oracle], "Model publisher already registered.");
        oracles.push(_oracle);
        registeredOracles[_oracle] = true;
        emit OracleRegistered(_oracle);
    }

    function registerModelPublisher(address _publisher) public onlyOwner {
        require(!registeredModelPublishers[_publisher], "Model publisher already registered.");
        modelPublishers.push(_publisher);
        registeredModelPublishers[_publisher] = true;
        emit ModelPublisherRegistered(_publisher);
    }

    function registerChargingStation(address _station) public onlyOwner {
    require(!registeredChargingStations[_station], "Charging station already registered.");
    chargingStations.push(_station);
    registeredChargingStations[_station] = true;
    reputation[_station] = 5e17;  // Initialize reputation to 5e17
    emit ChargingStationRegistered(_station);  // Ensure the event is emitted properly
    }

    function publishModel(string memory _modelCID, string memory _infoCID, uint maxRounds, uint requiredChargingStations) public {
        require(registeredModelPublishers[msg.sender], "Only Model publisher can invoke this");
        uint256 modelId = models.length;
        models.push(Model({
                modelId: modelId,
                modelCID: _modelCID,
                infoCID: _infoCID,
                publisher: msg.sender,
                chargingStations: new address [](0),             
                globalModelWeightsCID: "",
                currentRound: 0,
                maxRounds: maxRounds,
                requiredChargingStations: requiredChargingStations,
                registeredChargingStations: new address [](0),
           state: "selection"
            })
        );
       emit ModelPublished(modelId, _modelCID);
    }

    function joinChargingStationModel(uint _modelId) public payable {
        require(_modelId < models.length, "Model does not exist");
        require(registeredChargingStations[msg.sender], "Charging station not registered");
        Model storage model = models[_modelId];
        require(model.registeredChargingStations.length < model.requiredChargingStations, "Model has reached maximum capacity");
        require(!isInAddressArray(model.registeredChargingStations, msg.sender), "Charging station already joined the model");
        require(msg.value > 0, "Charging station deposit required");
        chargingStationDeposits[_modelId][msg.sender] = msg.value; // Store CS deposit
        model.registeredChargingStations.push(msg.sender);
        emit ChargingStationJoinedModel(_modelId, msg.sender,msg.value);
        
    }

    function makeDepositToStartTraining(uint _modelId) public payable {
       require(_modelId < models.length, "Model does not exist");
       Model storage model = models[_modelId];

       require(msg.sender == model.publisher, "Only model publisher can deposit");
       require(msg.value > 0, "Deposit amount must be greater than zero");
       require(modelDeposits[_modelId] == 0, "Deposit already made");

       modelDeposits[_modelId] = msg.value; // Store deposit for reward pool

       emit TrainingDepositMade(_modelId, msg.sender, msg.value);
    }

    function submitIntermediateModel(
        uint _modelId,
        bytes32 _hashIMCID,
        bytes[] memory signatures,
        address[] memory signers
    ) public {
        require(_modelId < models.length, "Invalid model ID");
        require(registeredChargingStations[msg.sender], "Only registered CS can submit");
        Model storage model = models[_modelId];
        require(isInAddressArray(model.registeredChargingStations, msg.sender), "Charging station not in the model");
        uint currentRound = model.currentRound;
        require(!updatesSubmitted[_modelId][currentRound][msg.sender], "Already submitted for this round");

        // Verify TS signatures before accepting IM submission
        require(signers.length == signatures.length, "Signers and signatures mismatch");
        require(signers.length >= threshold, "Not enough valid signatures");

        uint256 validSignatures = 0;
        for (uint256 i = 0; i < signers.length; i++) {
            require(_verifySignature(_hashIMCID, signatures[i], signers[i]), "Invalid signature");
            validSignatures++;
        }

        require(validSignatures >= threshold, "Threshold signatures not met");

        // Store submitted hash
        submittedHashes[_modelId][currentRound][msg.sender] = _hashIMCID;
        updatesSubmitted[_modelId][currentRound][msg.sender] = true;

        emit IntermediateModelSubmitted(_modelId, msg.sender, currentRound, _hashIMCID);
    }

    function _verifySignature(bytes32 _messageHash, bytes memory _signature, address _signer) internal pure returns (bool) {
        bytes32 ethSignedMessageHash = _toEthSignedMessageHash(_messageHash);
        return _recoverSigner(ethSignedMessageHash, _signature) == _signer;
    }

    function _toEthSignedMessageHash(bytes32 _hash) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", _hash));
    }

    function _recoverSigner(bytes32 _ethSignedMessageHash, bytes memory _signature) internal pure returns (address) {
        require(_signature.length == 65, "Invalid signature length");
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly {
            r := mload(add(_signature, 32))
            s := mload(add(_signature, 64))
            v := byte(0, mload(add(_signature, 96)))
        }
        return ecrecover(_ethSignedMessageHash, v, r, s);
    }


    function submitGlobalModel(
      uint _modelId,
      bytes32 _hashGMCID,
      bytes[] memory signatures,
      address[] memory signers
      ) public {
      require(_modelId < models.length, "Invalid model ID");
      require(registeredOracles[msg.sender], "Only registered oracles can submit");
      Model storage model = models[_modelId];

      uint currentRound = model.currentRound;
      require(!globalModelSubmitted[_modelId][currentRound], "Global model already submitted for this round");

      // Verify TS signatures before accepting global model submission
      require(signers.length == signatures.length, "Signers and signatures mismatch");
      require(signers.length >= threshold, "Not enough valid signatures");

      uint256 validSignatures = 0;
      for (uint256 i = 0; i < signers.length; i++) {
        require(_verifySignature(_hashGMCID, signatures[i], signers[i]), "Invalid signature");
        validSignatures++;
      }

      require(validSignatures >= threshold, "Threshold signatures not met");

      // Store submitted hash for global model
      globalModelHashes[_modelId][currentRound] = _hashGMCID;
      globalModelSubmitted[_modelId][currentRound] = true;

      emit GlobalModelSubmitted(_modelId, msg.sender, currentRound, _hashGMCID);
    }


    function isInAddressArray(address[] memory arr, address look) public pure returns (bool) {
        bool found = false;
        if (arr.length > 0) {
            for (uint i = 0; i < arr.length; i++) {
                if (arr[i] == look) {
                    found = true;
                    break;
                }
            }
        }
        return found;
    }

    function getBalance(address _addr) public view returns (uint256) {
        return balances[_addr];
    }

    function updateCSSelectionCount(
        uint256 _modelId, 
        address[] memory benignCSs, 
        bytes[] memory signatures, 
        address[] memory signers
    ) external {
        require(registeredOracles[msg.sender], "Only registered oracles can update CS selection");
        require(benignCSs.length > 0, "No benign CSs selected");
        require(signers.length == signatures.length, "Signers and signatures mismatch");
        require(signers.length >= threshold, "Not enough valid signatures");

        bytes32 messageHash = keccak256(abi.encodePacked(_modelId, benignCSs));
    
        uint256 validSignatures = 0;
        for (uint256 i = 0; i < signers.length; i++) {
          require(_verifySignature(messageHash, signatures[i], signers[i]), "Invalid signature");
          validSignatures++;
        }

        require(validSignatures >= threshold, "Threshold signatures not met");

        modelTrainingRounds[_modelId]++; // Increment total training rounds

        for (uint256 i = 0; i < benignCSs.length; i++) {
          address cs = benignCSs[i];
          csSelectionCount[cs]++; // Track how many times CS was selected
        }

        emit CSSelectionUpdated(_modelId, benignCSs);
    }

    function distributeReward(
        uint256 _modelId, 
        address[] memory CSs, 
        bytes[] memory signatures, 
        address[] memory signers
    ) external {
        require(registeredOracles[msg.sender], "Only registered oracles can distribute rewards");
        require(signers.length == signatures.length, "Signers and signatures mismatch");
        require(signers.length >= threshold, "Not enough valid signatures");

        bytes32 messageHash = keccak256(abi.encodePacked(_modelId, CSs));
    
        uint256 validSignatures = 0;
        for (uint256 i = 0; i < signers.length; i++) {
          require(_verifySignature(messageHash, signatures[i], signers[i]), "Invalid signature");
          validSignatures++;
        }

        require(validSignatures >= threshold, "Threshold signatures not met");

        uint256 totalReward = modelDeposits[_modelId];
        require(totalReward > 0, "No deposit to distribute");

        uint256 totalRounds = modelTrainingRounds[_modelId];
        require(totalRounds > 0, "No training rounds completed");

        uint256 totalSelections = 0;
        for (uint256 i = 0; i < CSs.length; i++) {
           totalSelections += csSelectionCount[CSs[i]];
        }
        require(totalSelections > 0, "No CSs participated");

        modelDeposits[_modelId] = 0; // Prevent reentrancy issues

        for (uint256 i = 0; i < CSs.length; i++) {
            address payable cs = payable(CSs[i]);
            uint256 reward = (totalReward * csSelectionCount[cs]) / totalSelections;

            (bool success, ) = cs.call{value: reward}("");
            require(success, "Transfer failed");

            // Update reputation using moving weighted mean
            uint256 oldRep = reputation[cs];
            uint256 newRep = ((oldRep * 9) + ((csSelectionCount[cs] * _scaleRep) / totalRounds)) / 10;
            reputation[cs] = newRep;

            emit ReputationUpdated(cs, newRep);
        }

        emit DepositReleased(_modelId, totalReward);
    }
}
