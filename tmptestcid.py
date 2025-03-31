from deval.utils import  cid_sha256_from_file, cid_sha256_from_str

testfname = "requirements.txt"
with open(testfname, 'r') as file:
    testfilecontent = file.read()
    cid_from_file = cid_sha256_from_file(testfname) 
    cid_from_text_self = cid_sha256_from_str(testfilecontent) 

    print("Test file content:")
    print(testfilecontent)
    print(f"CID from file: {cid_from_file} https://cid.ipfs.tech/#{cid_from_file}")
    print(f"CID from text self: {cid_from_text_self} https://cid.ipfs.tech/#{cid_from_text_self}")






 
