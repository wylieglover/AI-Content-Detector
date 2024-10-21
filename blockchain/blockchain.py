import hashlib

class Blockchain:
    def __init__(self):
        # Initialize any blockchain connection or parameters here if needed
        pass

    def blockchain_verification(self, content_data):
        # Compute the content hash
        content_hash = self.compute_content_hash(content_data)
        if content_hash is None:
            return {
                'verified': False,
                'message': 'Failed to compute content hash.'
            }

        # Simulate blockchain verification
        is_verified = self.check_blockchain_for_hash(content_hash)
        message = 'Content verified on blockchain.' if is_verified else 'Content not found on blockchain.'

        return {
            'verified': is_verified,
            'content_hash': content_hash,
            'message': message
        }

    def compute_content_hash(self, content_data):
        try:
            content_type = content_data['type']
            if content_type in ['image', 'video', 'audio']:
                # For binary data types, use the raw bytes
                raw_bytes = content_data.get('raw_bytes')
            elif content_type == 'text':
                # For text, encode the string
                raw_bytes = content_data['data'].encode('utf-8')
            else:
                return None

            # Compute SHA-256 hash
            sha256_hash = hashlib.sha256()
            sha256_hash.update(raw_bytes)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Error computing content hash: {e}")
            return None

    def check_blockchain_for_hash(self, content_hash):
        # Simulate by checking if the hash meets a condition
        # For example, if the hash starts with '00', we consider it verified
        return content_hash.startswith('00')