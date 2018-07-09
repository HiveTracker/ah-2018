using Bonsai.IO;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.IO.Ports;
using System.Linq;
using System.Reactive.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bonsai.HiveTracker
{
    public class BinaryCapture : Source<PulseData>
    {
        [TypeConverter(typeof(SerialPortNameConverter))]
        public string PortName { get; set; }

        public int BaudRate { get; set; }

        static float ReadCentroid(BinaryReader reader)
        {
            var pulseStart = reader.ReadUInt16() << 2;
            var pulseEnd = reader.ReadUInt16() << 2;
            return (pulseStart + pulseEnd) / (2f * 16);
        }

        public override IObservable<PulseData> Generate()
        {
            return Observable.Create<PulseData>((observer, cancellationToken) =>
            {
                return Task.Factory.StartNew(() =>
                {
                    using (var serialPort = new SerialPort(PortName, BaudRate))
                    {
                        serialPort.Open();
                        using (var reader = new BinaryReader(serialPort.BaseStream))
                        {
                            while (!cancellationToken.IsCancellationRequested)
                            {
                                while (reader.ReadByte() != byte.MaxValue) ;
                                if (reader.ReadByte() != byte.MaxValue) continue;

                                PulseData data;
                                var baseAxis = reader.ReadByte();
                                data.BaseId = (baseAxis >> 1) & 0x1;
                                data.Axis = baseAxis & 0x1;
                                data.Capture = new float[]
                                {
                                    ReadCentroid(reader),
                                    ReadCentroid(reader),
                                    ReadCentroid(reader),
                                    ReadCentroid(reader)
                                };
                                observer.OnNext(data);
                            }
                        }
                    }
                },
                cancellationToken,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default);
            });
        }
    }
}
