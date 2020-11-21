package BigDataAssignment2.A2T1;

import java.io.File;
import java.io.IOException;
import java.util.Set;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

public class StripesPart1 {

	private static final Logger LOG = Logger.getLogger(StripesPart1.class);

	/**
	 * This mapper class maps emits pair of form (word {{neighbor1, 1}, {neighbor2,
	 * 4}, ...}) for every line present in the input file
	 */
	public static class StripesOccurrenceMapper extends Mapper<LongWritable, Text, Text, MapWritable> {
		// map to store the frequency count of each neighbor
		private MapWritable neighborCountMap = new MapWritable();
		private Text word = new Text();

		@Override
		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

			String[] tokens = value.toString().split("\\s+");

			for (int i = 0; i < tokens.length; i++) {
				if (tokens[i].trim().length() > 0) {

					word.set(tokens[i].trim().toLowerCase());
					neighborCountMap.clear();

					// iterate over neighboring words in the sentence
					for (int j = 0; j < tokens.length; j++) {

						// continue if the neighbor word is not empty and not same as the key
						if (tokens[j].trim().length() > 0 && (tokens[i].trim() != tokens[j].trim())) {

							Text neighbor = new Text(tokens[j].trim().toLowerCase());

							// if the neighbor is already present then increase the frequency count by 1
							if (neighborCountMap.containsKey(neighbor)) {
								IntWritable count = (IntWritable) neighborCountMap.get(neighbor);
								count.set(count.get() + 1);
							} else {
								// else add the neighbor in the map with a count of 1
								neighborCountMap.put(neighbor, new IntWritable(1));
							}
						}
					}

					// emit the word pair count if the neighbor array is not empty
					if (!neighborCountMap.isEmpty())
						context.write(word, neighborCountMap);
				}
			}
		}
	}

	/**
	 * This reducer class reduces the output by combining the word pairs with same
	 * neighbor and generating the word-pair co-occurrence matrix using the strips
	 * approach
	 *
	 */
	public static class StripesReducer extends Reducer<Text, MapWritable, Text, MapWritable> {
		private MapWritable neighborFrequencyMap = new MapWritable();

		@Override
		protected void reduce(Text word, Iterable<MapWritable> neighborArrays, Context context)
				throws IOException, InterruptedException {

			neighborFrequencyMap.clear();

			// iterate over all the arrays containing the neighbor frequency count
			for (MapWritable neighborArray : neighborArrays) {

				Set<Writable> neighborWords = neighborArray.keySet();

				for (Writable neighbor : neighborWords) {
					IntWritable currentCount = (IntWritable) neighborArray.get(neighbor);

					// if the neighbor word is already present then increment the count by 1
					if (neighborFrequencyMap.containsKey(neighbor)) {
						IntWritable count = (IntWritable) neighborFrequencyMap.get(neighbor);
						count.set(count.get() + currentCount.get());
					} else {
						// if neighbor word in not present in map then add it in map with the count
						neighborFrequencyMap.put(neighbor, currentCount);
					}
				}
			}

			// emit the word and the neighbors and their frequency count as output
			context.write(word, neighborFrequencyMap);
		}
	}

	/**
	 * This is the main driver method for the map reduce program to calculate the
	 * word pair co-occurrence matrix using stripes approach
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		// log that the map
		LOG.setLevel(Level.INFO);
		LOG.info("Starting the Task 1.1 Stripes approach for Khushbu Patel, s3823274");

		Configuration conf = new Configuration();

		// set configuration and create a job for the mapreduce task
		Job job = Job.getInstance(conf, "Task 1.1 Stripes Approach");
		job.setJarByClass(StripesPart1.class);

		// set mapper, combiner and reducer classes
		job.setMapperClass(StripesOccurrenceMapper.class);
		job.setCombinerClass(StripesReducer.class);
		job.setReducerClass(StripesReducer.class);

		// set output key and value class
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(MapWritable.class);

		// delete output file path if already exists
		FileUtils.deleteDirectory(new File(args[1]));

		// set the input and output file path
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		// exit the program when the job is finished
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}

}
