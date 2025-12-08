import express, { Request, Response } from 'express';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, ScanCommand, DeleteCommand } from '@aws-sdk/lib-dynamodb';

const app = express();
const PORT = process.env.PORT || 3000;

// Initialize DynamoDB client
const client = new DynamoDBClient({ region: process.env.AWS_REGION || 'us-east-1' });
const docClient = DynamoDBDocumentClient.from(client);

app.use(express.json());

// GET /tracks
app.get('/tracks', (req: Request, res: Response) => {
  res.status(200).json({
    plannedTracks: ["Access control track"]
  });
});

// DELETE /reset
app.delete('/reset', async (req: Request, res: Response) => {
  try {
    const tableName = process.env.DYNAMODB_TABLE_NAME || 'PackageRegistry';
    
    const scanResult = await docClient.send(new ScanCommand({
      TableName: tableName,
    }));

    if (scanResult.Items && scanResult.Items.length > 0) {
      for (const item of scanResult.Items) {
        await docClient.send(new DeleteCommand({
          TableName: tableName,
          Key: { id: item.id },
        }));
      }
    }

    res.status(200).send('Registry is reset.');
  } catch (error) {
    console.error('Reset error:', error);
    res.status(500).json({ error: 'Failed to reset registry' });
  }
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
}

export default app;