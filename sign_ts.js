const { ethers } = require("ethers");
//console.log(ethers)
async function signCID(wallet, cid) {
    // Use solidityPackedKeccak256 to pack and hash the data
    const messageHash = ethers.solidityPackedKeccak256(
        ["string"], // specify the types (in this case, a single string)
        [cid]       // the corresponding CID value to be packed
    );

    // Sign the hashed message
   return await wallet.signMessage(ethers.toBeArray(messageHash)
);

}

async function generateSignatures() {
    const { Wallet } = ethers;
    const EVs = [...Array(5)].map(() => Wallet.createRandom());
    const cid = "CID_example";

    const signatures = await Promise.all(
        EVs.slice(0, 3).map(async (ev) => ({
            signer: ev.address,
            signature: await signCID(ev, cid),
        }))
    );

    console.log(JSON.stringify(signatures, null, 2));
}

generateSignatures();


/*"CID_example", [
    "0xd9c7d8a016f109d63cc1471417c6e8500e5e0f9a6be700da01ee74e2d005df5e7f785d2519ac0a19c80b5076f6c7b3d431965dc7d5066a92717d2e6c3688ea011b", 
    "0xd5bcfe00bce1d8a63021fda009f19f5118158642426f0bbad48e408abbf284327c70003f07e5edd5bf8e244c4a67e2cd09e0b85253a79782b0a25f3fb17ec6941b", 
    "0x636c88688781c93601b347c20665a34b186ba6744738e8db21f774c2f045cd4d357d812c300913b3ac361196b5ff058a86358b38223102b211d21027e6d506171c"
  ], [
    0x93D813680459890F7ea3a23d0b902b2440e0bD12, 
    0x88bd1412e7Cd615807fFECE1FCc36867a09467Ba, 
    0xDb457483942C91459abC035a26F74De784136291
  ]*/
  
