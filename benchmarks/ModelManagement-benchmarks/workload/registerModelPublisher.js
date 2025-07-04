/*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

'use strict';

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');
let fs = require('fs');
var isInit = false;
var myArray = [];


class StoreWorkload extends WorkloadModuleBase {

    /**
     * Initializes the parameters of the workload.
     */
    constructor() {
        super();
        this.txIndex = -1;
        this.contract = {};
    }



    /**
     * Generates simple workload.
     * @returns {{verb: String, args: Object[]}[]} Array of workload argument objects.
     */
     _generateWorkload() {

        
        let workload = [];
        let roundIndexes = [0,500,1500,2500];
        

        if(!isInit){
            isInit = true;
            var data = fs.readFileSync("./workload/scripts/publishersData.json");
            data= JSON.parse(data);
            data.forEach(item => {
                myArray.push(item);
            });
 
      }




        for(let i= 0 ; i <  this.roundArguments.txnPerBatch ; i++) {
            this.txIndex++; 
            let adr = myArray[this.txIndex+roundIndexes[this.roundArguments.num-1]].publisher ; 
            //console.log(adr)
            let args = {
                contract :this.roundArguments.contract,
                verb: 'registerModelPublisher',
                args: [adr],
                readOnly: false,
                // nonce: currentNonce, // Use the incremented nonce
                gasPrice: '200000' // Set a suitable gas pr  1000000000
            }
            // currentNonce++;

            if (this.isPrivate) {
                args.privacy = this.privacyOpts;
                args.privacy.sender = web3.eth.accounts.create();
                args.privacy.sender.nonce = 0;
            }
        
            workload.push(args);

        }

        return workload;
        
    }

    /**
     * Initialize the workload module with the given parameters.
     * @param {number} workerIndex The 0-based index of the worker instantiating the workload module.
     * @param {number} totalWorkers The total number of workers participating in the round.
     * @param {number} roundIndex The 0-based index of the currently executing round.
     * @param {Object} roundArguments The user-provided arguments for the round from the benchmark configuration file.
     * @param {BlockchainConnector} sutAdapter The adapter of the underlying SUT.
     * @param {Object} sutContext The custom context object provided by the SUT adapter.
     * @async
     */
    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {
        await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);
        if (!this.roundArguments.contract) {
            throw new Error('store - argument "contract" missing');
        }

        this.nodeUrl = sutContext.url;

        if(this.roundArguments.private) {
            this.isPrivate = true;
            this.privacyOpts = sutContext.privacy[this.roundArguments.private];
            this.privacyOpts['id'] = this.roundArguments.private;
        } else {
            this.isPrivate = false;
        }

        if(!this.roundArguments.txnPerBatch) {
            this.roundArguments.txnPerBatch = 1;
        }
    } 

   
    async submitTransaction() {
        let args =  this._generateWorkload();
        await this.sutAdapter.sendRequests(args);
    }
}

function createWorkloadModule() {
    return new StoreWorkload();
}

module.exports.createWorkloadModule = createWorkloadModule;