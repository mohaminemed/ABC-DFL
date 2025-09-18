// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

//RBAC-based access control
contract AccessManagement {
    
    address private admin;
    address  [] public adminsList ; 
    mapping(address => bool) public adminExist;
 

    address[] public _modelPublishers;
    address[] public _chargingStations;    

    address [] private oracleAdresses; 
    mapping (address => bool) private registeredOracles ; 

    constructor() {
        admin = msg.sender;
        adminsList.push(admin); 
        adminExist[admin]= true;
    }

    modifier onlyOwner() {
        require(adminExist[msg.sender] == true, "Only the contract owner can call this function.");
        _;
    }

    modifier onlyOracle() {
        require(registeredOracles[msg.sender] == true, "Only the oracle can call this function.");
        _;
    }

    function addOwner(address _newOwner) public onlyOwner {
        adminsList.push(_newOwner); 
        adminExist[_newOwner]= true;
    }

    function addOracle(address _address) public onlyOwner(){
        oracleAdresses.push(_address);
        registeredOracles[_address]=true;
    }

    function isOwner(address _address) public view returns (bool) {
        return adminExist[_address];
    }
}

