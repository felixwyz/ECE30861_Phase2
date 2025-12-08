import { Router, Request, Response } from 'express';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, ScanCommand, DeleteCommand } from '@aws-sdk/lib-dynamodb';

const router = Router();

// Initialize DynamoDB client
const dynamoDBClient = DynamoDBDocumentClient.from(new DynamoDBClient({
  region: process.env.AWS_REGION || 'us-east-1',
}));

/**
 * GET /tracks
 * Get the planned tracks
 */
router.get('/tracks', (req: Request, res: Response) => {
  try {
    const tracks = {
      plannedTracks: [
        "Access control track"
      ]
    };
    
    return res.status(200).json(tracks);
  } catch (error) {
    console.error('Error getting tracks:', error);
    return res.status(500).json({ error: 'Failed to get tracks' });
  }
});

/**
 * DELETE /reset
 * Reset the registry - delete all packages
 */
router.delete('/reset', async (req: Request, res: Response) => {
  try {
    const tableName = process.env.DYNAMODB_TABLE_NAME;
    if (!tableName) {
      return res.status(500).json({ error: 'Database configuration error' });
    }

    // Scan and delete all items
    const scanParams = {
      TableName: tableName,
    };

    const scanResult = await dynamoDBClient.send(new ScanCommand(scanParams));
    
    if (scanResult.Items && scanResult.Items.length > 0) {
      for (const item of scanResult.Items) {
        const deleteParams = {
          TableName: tableName,
          Key: {
            id: item.id,
          },
        };
        await dynamoDBClient.send(new DeleteCommand(deleteParams));
      }
    }

    return res.status(200).send('Registry is reset.');
  } catch (error) {
    console.error('Error resetting registry:', error);
    return res.status(500).json({ error: 'Failed to reset registry' });
  }
});

export default router;