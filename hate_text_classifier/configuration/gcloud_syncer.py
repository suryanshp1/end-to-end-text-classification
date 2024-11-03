from subprocess import Popen, PIPE


class GcloudSyncer:
    def sync_folder_to_gcloud(self, gcp_bucket_url, filepath, filename):
        process = Popen(['gsutil', 'cp', f'{filepath}/{filename}', f'gs://{gcp_bucket_url}/'], stdout=PIPE, stderr=PIPE, shell=True)
        process.communicate()

    def sync_folder_from_gcloud(self, gcp_bucket_url, filename, destination):
        process = Popen(['gsutil', 'cp', f'gs://{gcp_bucket_url}/{filename}', f'{destination}/{filename}'], stdout=PIPE, stderr=PIPE, shell=True)
        process.communicate()

